/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_GENERIC_SYCL_SYCL_POST_OPS_HPP
#define GPU_GENERIC_SYCL_SYCL_POST_OPS_HPP

#include "common/c_types_map.hpp"
#include "common/primitive_attr.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "xpu/sycl/memory_storage_base.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct ref_eltwise_fwd_t {
    ref_eltwise_fwd_t() = default;
    ref_eltwise_fwd_t(alg_kind_t alg, float alpha, float beta, float scale)
        : alg_(alg), alpha_(alpha), beta_(beta), scale_(scale) {
        using namespace alg_kind;
        assert(utils::one_of(alg_, eltwise_relu, eltwise_linear, eltwise_clip,
                eltwise_clip_v2, eltwise_hardswish, eltwise_gelu_tanh,
                eltwise_gelu_erf, eltwise_tanh, eltwise_logistic, eltwise_swish,
                eltwise_elu));
    }

    ref_eltwise_fwd_t(const post_ops_t::entry_t::eltwise_t &eltwise)
        : ref_eltwise_fwd_t(
                eltwise.alg, eltwise.alpha, eltwise.beta, eltwise.scale) {}

    float compute(float s) const {
        return compute(alg_, s, alpha_, beta_) * scale_;
    }

    template <int width>
    ::sycl::vec<float, width> compute(::sycl::vec<float, width> src_vec) const {
        ::sycl::vec<float, width> scale_vec(scale_);
        return compute(alg_, src_vec, alpha_, beta_) * scale_vec;
    }

    // Make `compute` functions static so that it can be used in an eltwise
    // primitive.
    template <int width>
    static ::sycl::vec<float, width> compute(alg_kind_t alg,
            ::sycl::vec<float, width> src_vec, float alpha, float beta) {
        using namespace alg_kind;
        using namespace math;

        constexpr ::sycl::vec<float, width> nan_vec(NAN);

        switch (alg) {
            case eltwise_relu: return relu_fwd(src_vec, alpha); ;
            case eltwise_linear: return linear_fwd(src_vec, alpha, beta);
            default: return nan_vec;
        }
    }

    static float compute(alg_kind_t alg, float s, float alpha, float beta) {
        using namespace alg_kind;
        using namespace math;

        float d = 0.f;
        switch (alg) {
            case eltwise_relu: d = relu_fwd(s, alpha); break;
            case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
            case eltwise_clip: d = clip_fwd(s, alpha, beta); break;
            case eltwise_clip_v2: d = clip_v2_fwd(s, alpha, beta); break;
            case eltwise_hardswish:
                d = dnnl::impl::math::hardswish_fwd(s, alpha, beta);
                break;
            case eltwise_gelu_tanh: d = gelu_tanh_fwd(s); break;
            case eltwise_gelu_erf: d = gelu_erf_fwd(s); break;
            case eltwise_tanh: d = tanh_fwd(s); break;
            case eltwise_logistic: d = logistic_fwd(s); break;
            case eltwise_swish:
                d = dnnl::impl::math::swish_fwd(s, alpha);
                break;
            case eltwise_elu: d = dnnl::impl::math::elu_fwd(s, alpha); break;
            default: d = ::sycl::nan(0u);
        }
        return d;
    }

private:
    alg_kind_t alg_;
    float alpha_;
    float beta_;
    float scale_;
};

struct ref_binary_op_t {
    ref_binary_op_t() = default;
    ref_binary_op_t(alg_kind_t alg, xpu::sycl::md_t src_md)
        : alg_(alg), src_md_(src_md) {
        using namespace alg_kind;
        assert(utils::one_of(alg_, binary_add, binary_div, binary_max,
                binary_min, binary_mul, binary_sub, binary_ge, binary_gt,
                binary_le, binary_lt, binary_eq, binary_ne));
    }

    ref_binary_op_t(const post_ops_t::entry_t::binary_t &binary)
        : ref_binary_op_t(binary.alg, xpu::sycl::md_t(&binary.src1_desc)) {}

    float load_and_compute(float s0, const xpu::sycl::in_memory_arg_t &src,
            dims_t offset) const { // TODO dims32_t
        memory_tensor_t src_mem(src, src_md_);
        float val = src_mem.load_md_bc(offset);
        return compute(s0, val);
    }

    float compute(float s0, float s1) const { return compute(alg_, s0, s1); }

    static float compute(alg_kind_t alg, float s0, float s1) {
        using namespace alg_kind;
        using namespace math;

        float d = 0.f;
        switch (alg) {
            case binary_add: d = s0 + s1; break;
            case binary_div: d = s0 / s1; break;
            case binary_min: d = ::sycl::min(s0, s1); break;
            case binary_max: d = ::sycl::max(s0, s1); break;
            case binary_mul: d = s0 * s1; break;
            case binary_sub: d = s0 - s1; break;
            case binary_ge: d = (float)((s0 >= s1) * -1); break;
            case binary_gt: d = (float)((s0 > s1) * -1); break;
            case binary_le: d = (float)((s0 <= s1) * -1); break;
            case binary_lt: d = (float)((s0 < s1) * -1); break;
            case binary_eq: d = (float)((s0 == s1) * -1); break;
            case binary_ne: d = (float)((s0 != s1) * -1); break;
            default: d = ::sycl::nan(0u);
        }
        return d;
    }

private:
    alg_kind_t alg_;
    xpu::sycl::md_t src_md_;
};

struct ref_sum_op_t {
    ref_sum_op_t() = default;
    ref_sum_op_t(float scale, float zeropoint)
        : scale_(scale), zeropoint_(zeropoint) {}

    float load_and_compute(float acc, const xpu::sycl::out_memory_arg_t &dst,
            dnnl::impl::data_type_t sum_dt_,
            dim_t offset) const { // TODO dims32_t
        memory_plain_t dst_mem(dst, sum_dt_);
        float val = dst_mem.load(offset);
        return compute(acc, val);
    }

    float compute(float acc, float dst) const {
        return acc + scale_ * (dst - zeropoint_);
    }

    template <int width>
    ::sycl::vec<float, width> compute(::sycl::vec<float, width> acc,
            ::sycl::vec<float, width> dst) const {
        const ::sycl::vec<float, width> scale_vec(scale_);
        const ::sycl::vec<float, width> zeropoint_vec(zeropoint_);
        return acc + scale_vec * (dst - zeropoint_vec);
    }

private:
    float scale_;
    float zeropoint_;
};

struct sycl_post_op_t {
    primitive_kind_t kind_;
    union {
        ref_binary_op_t binary_;
        ref_eltwise_fwd_t eltwise_;
        ref_sum_op_t sum_;
    };
};
struct post_op_input_args;

struct sycl_post_ops_t {
    // SYCL has a limitation on total size of kernel arguments.
    // This affects number of post ops, e.g. binary post op (which is not yet
    // implemented) contains xpu::sycl::md_t which is large enough to limit
    // the number of post ops.
    static constexpr int max_post_ops = 5;

    sycl_post_ops_t() = default;
    sycl_post_ops_t(const primitive_attr_t *attr,
            dnnl::impl::data_type_t dst_dt = dnnl_data_type_undef) {
        using namespace primitive_kind;

        const auto &attr_po = attr->post_ops_;
        assert(attr_po.len() <= max_post_ops);

        for (auto i = 0; i < attr_po.len(); ++i) {
            if (attr_po.contain(sum, i)) {
                ops_[i].kind_ = sum;
                ops_[i].sum_ = ref_sum_op_t(attr_po.entry_[i].sum.scale,
                        attr_po.entry_[i].sum.zero_point);
                sum_dt_ = attr_po.entry_[i].sum.dt == dnnl_data_type_undef
                        ? dst_dt
                        : attr_po.entry_[i].sum.dt;
            } else if (attr_po.contain(eltwise, i)) {
                ops_[i].kind_ = eltwise;
                ops_[i].eltwise_ = ref_eltwise_fwd_t(attr_po.entry_[i].eltwise);
            } else if (attr_po.contain(binary, i)) {
                ops_[i].kind_ = binary;
                ops_[i].binary_ = ref_binary_op_t(attr_po.entry_[i].binary);
            }
        }
        n_post_ops_ = attr_po.len();
    }

    inline float apply(float acc, const xpu::sycl::out_memory_arg_t &dst,
            dim_t dst_offset, const post_op_input_args &po_args,
            dims_t src_offset) const;
    inline float apply(float acc, const post_op_input_args &po_args,
            dims_t src_offset) const;
    inline float apply(float acc, const xpu::sycl::out_memory_arg_t &dst,
            dim_t dst_offset) const;

    inline int get_post_op() const { return n_post_ops_; }

    inline primitive_kind_t get_post_op_kind(int i) const {
        return ops_[i].kind_;
    }

    inline ref_binary_op_t get_binary_post_op(int i) const {
        return ops_[i].binary_;
    }

    //there can be at most one sum type
    dnnl::impl::data_type_t sum_dt_;

private:
    sycl_post_op_t ops_[max_post_ops];
    // Indicates the actual number of post ops.
    int n_post_ops_;
};

struct post_op_input_args {
    post_op_input_args(::sycl::handler &cgh, const exec_ctx_t &ctx)
        : args_ {CTX_IN_SYCL_KERNEL_MEMORY(
                         (DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1)),
                CTX_IN_SYCL_KERNEL_MEMORY(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1)),
                CTX_IN_SYCL_KERNEL_MEMORY(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(2) | DNNL_ARG_SRC_1)),
                CTX_IN_SYCL_KERNEL_MEMORY(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | DNNL_ARG_SRC_1)),
                CTX_IN_SYCL_KERNEL_MEMORY(
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(4) | DNNL_ARG_SRC_1))} {
    }

    xpu::sycl::in_memory_arg_t args_[sycl_post_ops_t::max_post_ops];
};

float sycl_post_ops_t::apply(float acc, const xpu::sycl::out_memory_arg_t &dst,
        dim_t dst_offset, const post_op_input_args &po_args,
        dims_t src_offset) const {
    using namespace primitive_kind;

    for (auto i = 0; i < n_post_ops_; ++i) {
        switch (ops_[i].kind_) {
            case eltwise: acc = ops_[i].eltwise_.compute(acc); break;
            case binary:
                acc = ops_[i].binary_.load_and_compute(
                        acc, po_args.args_[i], src_offset);
                break;
            case sum:
                acc = ops_[i].sum_.load_and_compute(
                        acc, dst, sum_dt_, dst_offset);
                break;
            default: acc = ::sycl::nan(0u);
        }
    }
    return acc;
}

float sycl_post_ops_t::apply(
        float acc, const post_op_input_args &po_args, dims_t src_offset) const {
    using namespace primitive_kind;

    for (auto i = 0; i < n_post_ops_; ++i) {
        switch (ops_[i].kind_) {
            case eltwise: acc = ops_[i].eltwise_.compute(acc); break;
            case binary:
                acc = ops_[i].binary_.load_and_compute(
                        acc, po_args.args_[i], src_offset);
                break;
            default: acc = ::sycl::nan(0u);
        }
    }
    return acc;
}

float sycl_post_ops_t::apply(float acc, const xpu::sycl::out_memory_arg_t &dst,
        dim_t dst_offset) const {
    using namespace primitive_kind;

    for (auto i = 0; i < n_post_ops_; ++i) {
        switch (ops_[i].kind_) {
            case eltwise: acc = ops_[i].eltwise_.compute(acc); break;
            case sum:
                acc = ops_[i].sum_.load_and_compute(
                        acc, dst, sum_dt_, dst_offset);
                break;
            default: acc = ::sycl::nan(0u);
        }
    }
    return acc;
}

CHECK_SYCL_KERNEL_ARG_TYPE(ref_binary_op_t);
CHECK_SYCL_KERNEL_ARG_TYPE(ref_eltwise_fwd_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_post_op_t);
CHECK_SYCL_KERNEL_ARG_TYPE(sycl_post_ops_t);

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif