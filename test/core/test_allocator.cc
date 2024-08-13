#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/unary.h"

#include "test.h"

namespace infini
{
    TEST(Allocator, testAlloc)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c
        size_t offsetA = allocator.alloc(a->getBytes()); // 48
        size_t offsetB = allocator.alloc(b->getBytes()); // 96
        size_t offsetC = allocator.alloc(c->getBytes()); // 144
        // free b, then allocate d
        allocator.free(offsetB, b->getBytes());          // 96   48
        size_t offsetD = allocator.alloc(d->getBytes()); // 144
        // expected to be a->d->c
        EXPECT_EQ(offsetB, offsetD);
        ASSERT_FALSE(offsetA == 0 && offsetB == 0 && offsetC == 0 && offsetD == 0);
    }

    TEST(Allocator, testAllocWithEndFreeBlock)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime); // 12 * 4
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d =
            make_ref<TensorObj>(Shape{2, 2, 2, 3}, DataType::Float32, runtime); //  4 * 24
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c
        allocator.alloc(a->getBytes()); // 48   96
        allocator.alloc(b->getBytes()); // 48   96
        size_t offsetC = allocator.alloc(c->getBytes()); // 48   144
        allocator.info();
        // free c, then allocate d
        allocator.free(offsetC, c->getBytes()); // -48   96    48
        size_t offsetD = allocator.alloc(d->getBytes()); // +96 192
        allocator.info();
        // expected to be a->b->d, with no free block between b and c
        EXPECT_EQ(offsetC, offsetD);
    }

    TEST(Allocator, testGetPtr)
    {
        Shape shape = Shape{1, 2, 2, 3};
        Runtime runtime = NativeCpuRuntimeObj::getInstance();
        Tensor a = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor b = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor c = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Tensor d = make_ref<TensorObj>(shape, DataType::Float32, runtime);
        Allocator allocator = Allocator(runtime);
        // allocate a->b->c->d
        allocator.alloc(a->getBytes());
        allocator.alloc(b->getBytes());
        allocator.alloc(c->getBytes());
        allocator.alloc(d->getBytes());
        // multiple calls to the getPtr() function should return the same pointer
        void *ptr1 = allocator.getPtr();
        void *ptr2 = allocator.getPtr();
        EXPECT_EQ(ptr1, ptr2);
    }

} // namespace infini
