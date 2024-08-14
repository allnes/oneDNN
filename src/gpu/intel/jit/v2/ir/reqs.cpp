/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/intel/jit/v2/ir/reqs.hpp"

#include "gpu/intel/jit/ir/linear_expr.hpp"
#include "gpu/intel/jit/pass/simplify.hpp"

#include <iostream>

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {
namespace v2 {

bool is_a_mod_b_eq_0(const expr_t &e, expr_t &a, int &b) {
    auto *eq_op = e.as_ptr<binary_op_t>();
    if (!eq_op || eq_op->op_kind != op_kind_t::_eq) return false;
    if (!is_zero(eq_op->b)) return false;
    auto *mod_op = eq_op->a.as_ptr<binary_op_t>();
    if (!mod_op || mod_op->op_kind != op_kind_t::_mod) return false;
    if (!is_const(mod_op->b)) return false;
    a = mod_op->a;
    b = to_cpp<int>(mod_op->b);
    return true;
}

class linear_cmp_simplifier_t : public ir_mutator_t {
public:
    object_t _mutate(const binary_op_t &obj) override {
        if (!is_cmp_op(obj.op_kind)) return ir_mutator_t::_mutate(obj);

        if (!is_const(obj.b)) return obj;
        int a_div = linear_max_pow2_divisor(obj.a);
        int b_div = to_cpp<int>(obj.b);
        int factor = math::gcd(a_div, b_div);
        if (factor == 1) return obj;

        auto a = linear_div(obj.a, factor);
        auto b = b_div / factor;
        return binary_op_t::make(obj.op_kind, a, b);
    }
};

expr_t simplify_expr(const expr_t &_e) {
    expr_t a;
    int b;
    if (is_a_mod_b_eq_0(_e, a, b)) {
        return simplify_linear_mod(simplify_expr(a), b) == 0;
    }
    auto e = _e;
    for (int i = 0; i < 2; i++) {
        e = simplify_rewrite(e);
        e = linear_cmp_simplifier_t().mutate(e);
        e = const_fold(e);
    }
    return e;
}

req_expr_t to_req_expr(const expr_t &e);

void prb_reqs_t::add(const expr_t &_e) {
    auto e = simplify_expr(_e);
    if (auto *imm = e.as_ptr<bool_imm_t>()) {
        if (imm->value) return;
        ir_error_not_expected() << _e;
    }
    add_if_not_found(to_req_expr(e));
}

void prb_reqs_t::add(const prb_reqs_t &other) {
    for (auto &r : other.reqs_)
        add_if_not_found(r.expr);
}

void prb_reqs_t::add(const prb_tile_t &tile) {
    for (auto &d : tile) {
        set(d, tile[d]);
    }
}

void prb_reqs_t::set(const prb_dim_t &dim, int value) {
    add(size_var(dim) == value);
}

void prb_reqs_t::add_if_not_found(const req_expr_t &re) {
    auto e = re.to_ir();
    for (auto &r : reqs_) {
        if (r.expr.to_ir().is_equal(e)) return;
    }
    reqs_.emplace_back(re);
}

prover_t prb_reqs_t::prover(bool enable) {
    if (!enable) return prover_t();
    return prover_t(this);
}

bool prb_reqs_t::fits(const prb_tile_t &sizes) const {
    for (auto &r : reqs_) {
        ir_check(r.fits(sizes));
    }
    return true;
}

void prb_reqs_t::stringify(std::ostream &out) const {
    if (reqs_.empty()) {
        out << "x";
        return;
    }
    bool is_first = true;
    for (auto &r : reqs_) {
        if (!is_first) out << ":";
        r.stringify(out);
        is_first = false;
    }
}

void prb_reqs_t::parse(std::istream &in) {
    reqs_.clear();
    auto s = stream_parse<std::string>(in);
    if (s == "x") return;
    auto parts = gpu_utils::split(s, ":");
    for (auto &p : parts) {
        auto r = jit::parse<req_t>(p);
        reqs_.push_back(r);
    }
}

std::string prb_reqs_t::str() const {
    std::ostringstream oss;
    bool is_first = true;
    for (auto &r : reqs_) {
        if (!is_first) oss << "\n";
        oss << r.expr.to_ir();
        is_first = false;
    }
    return oss.str();
}

class req_int_imm_t : public req_expr_impl_t {
public:
    req_int_imm_t(int64_t value) : value(value) {}
    static req_expr_t make(int64_t value) {
        return req_expr_t(new req_int_imm_t(value));
    }
    ir_type_id_t expr_kind() const override { return ir_type_id_t::int_imm_t; }
    int64_t to_int(const prb_tile_t &sizes) const override { return value; }
    expr_t to_ir() const override { return int_imm_t::make(value); }

    int64_t value;
};

class req_const_var_t : public req_expr_impl_t {
public:
    req_const_var_t(const prb_dim_t &dim) : dim(dim) {
        ir_assert(!dim.is_undef());
    }

    static req_expr_t make(const prb_dim_t &dim) {
        return req_expr_t(new req_const_var_t(dim));
    }

    ir_type_id_t expr_kind() const override {
        return ir_type_id_t::const_var_t;
    }

    int64_t to_int(const prb_tile_t &sizes) const override {
        return sizes.at(dim);
    }

    expr_t to_ir() const override { return size_var(dim); }

    prb_dim_t dim;
};

class req_unary_op_t : public req_expr_impl_t {
public:
    req_unary_op_t(op_kind_t op_kind, const req_expr_t &a)
        : op_kind(op_kind), a(a) {}

    static req_expr_t make(op_kind_t op_kind, const req_expr_t &a) {
        return req_expr_t(new req_unary_op_t(op_kind, a));
    }

    ir_type_id_t expr_kind() const override { return ir_type_id_t::unary_op_t; }

    int64_t to_int(const prb_tile_t &sizes) const override {
        switch (op_kind) {
            case op_kind_t::_minus: return -a.to_int(sizes);
            default: ir_error_not_expected();
        }
        return 0;
    }

    expr_t to_ir() const override {
        return unary_op_t::make(op_kind, a.to_ir());
    }

    op_kind_t op_kind;
    req_expr_t a;
};

class req_binary_op_t : public req_expr_impl_t {
public:
    req_binary_op_t(op_kind_t op_kind, const req_expr_t &a, const req_expr_t &b)
        : op_kind(op_kind), a(a), b(b) {}

    static req_expr_t make(
            op_kind_t op_kind, const req_expr_t &a, const req_expr_t &b) {
        return req_expr_t(new req_binary_op_t(op_kind, a, b));
    }

    ir_type_id_t expr_kind() const override {
        return ir_type_id_t::binary_op_t;
    }

    int64_t to_int(const prb_tile_t &sizes) const override {
        int64_t a_value = a.to_int(sizes);
        int64_t b_value = b.to_int(sizes);
        switch (op_kind) {
            case op_kind_t::_add: return a_value + b_value;
            case op_kind_t::_sub: return a_value - b_value;
            case op_kind_t::_mul: return a_value * b_value;
            case op_kind_t::_mod: return a_value % b_value;
            default: ir_error_not_expected();
        }
        return 0;
    }

    expr_t to_ir() const override {
        return binary_op_t::make(op_kind, a.to_ir(), b.to_ir());
    }

    op_kind_t op_kind;
    req_expr_t a;
    req_expr_t b;
};

req_expr_t to_req_expr(const expr_t &e) {
    if (auto *ptr = e.as_ptr<int_imm_t>()) {
        return req_int_imm_t::make(ptr->value);
    }
    if (e.is<const_var_t>()) {
        return req_const_var_t::make(size_to_prb_dim(e));
    }
    if (auto *ptr = e.as_ptr<unary_op_t>()) {
        auto a = to_req_expr(ptr->a);
        return req_unary_op_t::make(ptr->op_kind, a);
    }
    if (auto *ptr = e.as_ptr<binary_op_t>()) {
        auto a = to_req_expr(ptr->a);
        auto b = to_req_expr(ptr->b);
        return req_binary_op_t::make(ptr->op_kind, a, b);
    }
    ir_error_not_expected() << e;
    return 0;
}

bool prb_reqs_t::req_t::fits(const prb_tile_t &sizes) const {
    if (auto *op = expr.as_ptr<req_binary_op_t>()) {
        int64_t a = op->a.to_int(sizes);
        int64_t b = op->b.to_int(sizes);
        bool ret = false;
        switch (op->op_kind) {
            case op_kind_t::_eq: ret = (a == b); break;
            case op_kind_t::_ge: ret = (a >= b); break;
            case op_kind_t::_gt: ret = (a > b); break;
            case op_kind_t::_le: ret = (a <= b); break;
            case op_kind_t::_lt: ret = (a < b); break;
            default: ir_error_not_expected();
        }
        ir_check(ret) << "Requirement is not satisfied: " << expr.to_ir()
                      << " evaluates to " << a << " " << op->op_kind << " "
                      << b;
        return true;
    }
    ir_error_not_expected() << expr.to_ir();
    return false;
}

bool prb_reqs_t::req_t::can_prove(const expr_t &expr_to_prove) const {
    expr_t lhs_a, rhs_a;
    int lhs_b, rhs_b;
    auto lhs = expr.to_ir();
    auto rhs = expr_to_prove;
    if (lhs.is_equal(rhs)) return true;
    if (!is_a_mod_b_eq_0(lhs, lhs_a, lhs_b)) return false;
    if (!is_a_mod_b_eq_0(rhs, rhs_a, rhs_b)) return false;
    auto lhs_dim = size_to_prb_dim(lhs_a);
    auto rhs_dim = size_to_prb_dim(rhs_a);
    if (lhs_dim.is_undef() || rhs_dim.is_undef() || lhs_dim != rhs_dim)
        return false;
    if (lhs_b % rhs_b == 0) return true;
    return false;
}

bool to_dims_mod_c(const req_expr_t &re, std::vector<prb_dim_t> &dims, int &c) {
    expr_t e = re.to_ir();
    expr_t a;
    if (!is_a_mod_b_eq_0(e, a, c)) return false;
    auto mul_args = op_split(op_kind_t::_mul, a);
    dims.clear();
    for (auto &ma : mul_args) {
        auto dim = size_to_prb_dim(ma);
        if (dim.is_undef()) return false;
        dims.push_back(dim);
    }
    return true;
}

void prb_reqs_t::simplify() {
    dim_map_t<prb_dim_t, int> mod_info;
    mod_info.fill_missing(1);
    for (auto &r : reqs_) {
        std::vector<prb_dim_t> dims;
        int c;
        if (to_dims_mod_c(r.expr, dims, c)) {
            if (dims.size() == 1) {
                int &f = mod_info[dims[0]];
                f = std::max(f, c);
            }
        }
    }
    std::vector<req_t> new_reqs;
    for (auto &r : reqs_) {
        std::vector<prb_dim_t> dims;
        int c;
        if (to_dims_mod_c(r.expr, dims, c)) {
            int f = 1;
            for (auto &d : dims)
                f *= mod_info.at(d);
            if (f % c == 0) continue;
        }
        new_reqs.push_back(r);
    }
    for (auto &d : mod_info) {
        if (mod_info[d] == 1) continue;
        new_reqs.emplace_back(to_req_expr((size_var(d) % mod_info[d]) == 0));
    }
    reqs_ = new_reqs;
}

bool prb_reqs_t::can_prove(const expr_t &_e) const {
    auto e = simplify_expr(_e);
    if (auto *imm = e.as_ptr<bool_imm_t>()) { return imm->value; }
    for (auto &r : reqs_) {
        if (r.can_prove(e)) return true;
    }
    return false;
}

bool prb_reqs_t::get_value(const prb_dim_t &dim, int &value) const {
    auto var = size_var(dim);
    for (auto &r : reqs_) {
        auto e = r.expr.to_ir();
        if (auto *op = e.as_ptr<binary_op_t>()) {
            if (op->op_kind == op_kind_t::_eq && op->a.is_equal(var)
                    && is_const(op->b)) {
                value = to_cpp<int>(op->b);
                return true;
            }
        }
    }
    return false;
}

bool prb_reqs_t::is_equal(const prb_dim_t &dim, int value) const {
    int dim_value;
    return get_value(dim, dim_value) && dim_value == value;
}

bool prb_reqs_t::implies(const prb_reqs_t &other) const {
    for (auto &req : other.reqs_) {
        if (!can_prove(req.expr.to_ir())) return false;
    }
    return true;
}

expr_t prb_reqs_t::to_expr(const prb_dim_t &dim) const {
    int dim_value;
    if (get_value(dim, dim_value)) return dim_value;
    return size_var(dim);
}

void stringify_var_mul(std::ostream &out, const expr_t &e) {
    auto args = op_split(op_kind_t::_mul, e);
    bool is_first = true;
    for (auto &a : args) {
        auto dim = size_to_prb_dim(a);
        if (dim.is_undef()) ir_error_not_expected() << a;
        if (!is_first) out << "*";
        out << dim.str();
        is_first = false;
    }
}

void prb_reqs_t::req_t::stringify(std::ostream &out) const {
    auto e = expr.to_ir();
    ir_assert(e.is<binary_op_t>());
    auto &op = e.as<binary_op_t>();
    switch (op.op_kind) {
        case op_kind_t::_le:
        case op_kind_t::_ge:
        case op_kind_t::_eq: {
            ir_assert(is_const(op.b));
            auto *mod_op = op.a.as_ptr<binary_op_t>();
            if (mod_op && mod_op->op_kind == op_kind_t::_mod) {
                ir_assert(is_const(mod_op->b)) << e;
                stringify_var_mul(out, mod_op->a);
                out << "%" << mod_op->b;
            } else {
                stringify_var_mul(out, op.a);
            }
            out << op.op_kind;
            out << op.b;
            break;
        }
        default: ir_error_not_expected() << e;
    }
}

expr_t parse_var_mul(const std::string &s) {
    auto parts = gpu_utils::split(s, "*");
    std::vector<expr_t> args;
    for (auto &p : parts) {
        auto dim = prb_dim_t::from_name(p);
        args.push_back(size_var(dim));
    }
    return op_combine(op_kind_t::_mul, args);
}

void prb_reqs_t::req_t::parse(std::istream &in) {
    std::string s;
    in >> s;
    for (op_kind_t op : {op_kind_t::_le, op_kind_t::_ge, op_kind_t::_eq}) {
        auto s_op = to_string(op);
        auto pos = s.find(s_op);
        if (pos == std::string::npos) continue;
        auto s_lhs = s.substr(0, pos);
        auto s_rhs = s.substr(pos + s_op.length());
        auto mod_pos = s_lhs.find("%");
        expr_t lhs;
        if (mod_pos != std::string::npos) {
            auto s_mod_lhs = s_lhs.substr(0, mod_pos);
            auto s_mod_rhs = s_lhs.substr(mod_pos + 1);
            auto mod_lhs = parse_var_mul(s_mod_lhs);
            auto mod_rhs = std::stoi(s_mod_rhs);
            lhs = mod_lhs % mod_rhs;
        } else {
            lhs = parse_var_mul(s_lhs);
        }
        auto rhs = std::stoi(s_rhs);
        auto e = binary_op_t::make(op, lhs, rhs);
        operator=(to_req_expr(e));
        return;
    }
    ir_error_not_expected() << s;
}

std::string prb_reqs_t::req_t::str() const {
    return expr.to_ir().str();
}

const prover_t &prover_t::instance() {
    static prover_t _instance;
    return _instance;
}

bool prover_t::require(const expr_t &_e) const {
    auto e = simplify_expr(_e);
    if (auto *imm = e.as_ptr<bool_imm_t>()) return imm->value;

    if (!parent_) return false;
    parent_->add_if_not_found(to_req_expr(e));
    return true;
}

} // namespace v2
} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
