#include <stdlib.h>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "common.h"
#include "khash.h"


/* The value indicating missingess */
#define NIL -1


/* Define the hashtable structs
 * These map the items to counters for each dtype */

/* int64 dtypes */
KHASH_MAP_INIT_INT64(int64, npy_intp)

/* object dtype */
CRICK_INLINE int pyobject_cmp(PyObject* a, PyObject* b) {
	int result = PyObject_RichCompareBool(a, b, Py_EQ);
	if (result < 0) {
		PyErr_Clear();
		return 0;
	}
	return result;
}

KHASH_INIT(object, PyObject*, npy_intp, 1, PyObject_Hash, pyobject_cmp)


/* Generic SpaceSaving struct, used for casting to access the consistent fields
 * */
#define SPSV_HEAD      \
    npy_intp capacity; \
    npy_intp size;     \
    npy_intp head;

typedef struct {
    SPSV_HEAD
} spsv_t;

/*
 * BEGIN LOOP
 */

typedef struct {
    npy_int64 item;
    npy_int64 count;
    npy_int64 error;
} counter_int64_t;


typedef struct {
    npy_intp next;
    npy_intp prev;
    counter_int64_t counter;
} node_int64_t;


typedef struct {
    SPSV_HEAD
    node_int64_t *list;
    khash_t(int64) *hashmap;
} spsv_int64_t;


CRICK_INLINE spsv_int64_t *spsv_int64_new(int capacity) {
    spsv_int64_t *T = (spsv_int64_t *)malloc(sizeof(*T));
    if (T == NULL)
        goto fail;

    T->list = (node_int64_t *)malloc(capacity * sizeof(node_int64_t));
    if (T->list == NULL)
        goto fail;

    T->capacity = capacity;
    T->size = 0;
    T->head = NIL;
    T->hashmap = kh_init(int64);

    return T;

fail:
    if (T->list != NULL) free(T->list);
    if (T != NULL) free(T);
    return NULL;
}


CRICK_INLINE void spsv_int64_free(spsv_int64_t *T) {
    kh_destroy(int64, T->hashmap);
    free(T->list);
    free(T);
}


CRICK_INLINE int counter_int64_ge(counter_int64_t c1,
                                     counter_int64_t c2,
                                     npy_int64 offset) {
    npy_int64 count = c2.count + offset;
    npy_int64 error = c2.error + offset;
    return (c1.count > count || (c1.count == count && c1.error <= error));
}


CRICK_INLINE void spsv_int64_counter_insert(spsv_int64_t *T,
                                                npy_intp c, npy_intp prev) {
    npy_intp tail = T->list[T->head].prev;
    while(1) {
        if (counter_int64_ge(T->list[prev].counter,
                                T->list[c].counter, 0))
            break;
        prev = T->list[prev].prev;
        if (prev == tail) {
            T->head = c;
            break;
        }
    }
    T->list[c].next = T->list[prev].next;
    T->list[c].prev = prev;
    T->list[T->list[prev].next].prev = c;
    T->list[prev].next = c;
}


CRICK_INLINE npy_intp spsv_int64_counter_new(spsv_int64_t *T,
                                                npy_int64 item, npy_int64 count,
                                                npy_int64 error) {
    npy_intp c = T->size;
    T->size++;

    T->list[c].counter.item = item;
    T->list[c].counter.count = count;
    T->list[c].counter.error = error;

    if (T->head == NIL) {
        T->head = c;
        T->list[c].prev = c;
        T->list[c].next = c;
    }
    else {
        npy_intp tail = T->list[T->head].prev;
        spsv_int64_counter_insert(T, c, tail);
    }
    return c;
}


CRICK_INLINE void spsv_int64_rebalance(spsv_int64_t *T, npy_intp index) {
    npy_intp prev;
    if (T->head == index) {
        /* Counts can only increase */
        return;
    }
    prev = T->list[index].prev;

    if (counter_int64_ge(T->list[prev].counter,
                            T->list[index].counter, 0))
        return;

    /* Counter needs to be moved. Remove then insert. */
    T->list[T->list[index].next].prev = prev;
    T->list[prev].next = T->list[index].next;
    spsv_int64_counter_insert(T, index, prev);
}


CRICK_INLINE int spsv_int64_swap(spsv_int64_t *T, npy_intp index,
                                    npy_int64 item, npy_int64 count,
                                    npy_int64 error) {
    /* Remove the old item from the hashmap */
    npy_int64 old_item = T->list[index].counter.item;
    khiter_t iter = kh_get(int64, T->hashmap, old_item);
    if (iter == kh_end(T->hashmap)) return -1;


    kh_del(int64, T->hashmap, iter);


    T->list[index].counter.item = item;
    T->list[index].counter.error = error;
    T->list[index].counter.count = count;
    spsv_int64_rebalance(T, index);
    return 0;
}


CRICK_INLINE int spsv_int64_add(spsv_int64_t *T, npy_int64 item,
                                   npy_int64 count) {
    int absent;
    npy_intp index;

    /* Get the pointer to the bucket */
    khiter_t iter = kh_put(int64, T->hashmap, item, &absent);
    /* If the key is an object, we need to check for hash failure */
    if (absent > 0) {
        /* New item */
        if (T->size == T->capacity) {
            /* we're full, replace the min counter */
            index = T->list[T->head].prev;
            if (spsv_int64_swap(T, index, item,
                                   T->list[index].counter.count + 1,
                                   T->list[index].counter.count))
                return -1;
        } else {
            /* Not full, allocate a new counter */
            index = spsv_int64_counter_new(T, item, count, 0);
        }
        kh_val(T->hashmap, iter) = index;
    }
    else if (absent == 0) {
        /* The counter exists, just update it */
        index = kh_val(T->hashmap, iter);
        T->list[index].counter.count += count;
        spsv_int64_rebalance(T, index);
    }
    else {
        PyErr_NoMemory();
        return -1;
    }
    return 1;
}


CRICK_INLINE int spsv_int64_set_state(spsv_int64_t *T,
                                         counter_int64_t *counters,
                                         npy_intp size) {
    int i, absent;
    npy_intp index;
    if (size > T->capacity) {
        PyErr_SetString(PyExc_ValueError,
                        "deserialization failed, size > capacity");
        return -1;
    }
    for (i=0; i < size; i++) {
        counter_int64_t c = counters[i];
        /* Get the pointer to the bucket */
        khiter_t iter = kh_put(int64, T->hashmap, c.item, &absent);
        /* If the key is an object, we need to check for hash failure */
        if (absent > 0) {
            index = spsv_int64_counter_new(T, c.item, c.count, c.error);
            kh_val(T->hashmap, iter) = index;
        }
        else if (absent == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "deserialization failed, duplicate items found");
            return -1;
        }
        else {
            PyErr_NoMemory();
            return -1;
        }
    }
    return 1;
}


CRICK_INLINE int spsv_int64_merge(spsv_int64_t *T1, spsv_int64_t *T2) {
    npy_int64 m1, m2;
    npy_intp i1, i2;
    int i;
    /* Nothing to do */
    if (T2->size == 0) return 0;

    /* Get the minimum counts from each */
    if (T1->size < T1->capacity)
        m1 = 0;
    else
        m1 = T1->list[T1->list[T1->head].prev].counter.count;

    if (T2->size < T2->capacity)
        m2 = 0;
    else
        m2 = T2->list[T2->list[T2->head].prev].counter.count;

    /* Iterate through T1, updating it inplace */
    for (i1 = 0; i1 < T1->size; i1++) {
        khiter_t iter = kh_get(int64, T2->hashmap, T1->list[i1].counter.item);


        if (iter != kh_end(T2->hashmap)) {
            /* item is in both T1 and T2 */
            i2 = kh_val(T2->hashmap, iter);
            T1->list[i1].counter.count += T2->list[i2].counter.count;
            T1->list[i1].counter.error += T2->list[i2].counter.error;
        }
        else {
            /* item is in only T1 */
            T1->list[i1].counter.count += m2;
            T1->list[i1].counter.error += m2;
        }
        spsv_int64_rebalance(T1, i1);
    }

    /* Iterate through T2, adding in any missing items */
    i2 = T2->head;
    for (i = 0; i < T2->size; i++) {
        int absent;
        counter_int64_t c2 = T2->list[i2].counter;
        khiter_t iter = kh_put(int64, T1->hashmap, c2.item, &absent);

        if (absent > 0) {
            /* Item isn't in T1, maybe add it */
            if (T1->size == T1->capacity) {
                i1 = T1->list[T1->head].prev;

                /* If all counts in T1 are >= T2 + m1 then we're done here */
                if (counter_int64_ge(T1->list[i1].counter, c2, m1)) break;

                if (spsv_int64_swap(T1, i1, c2.item, c2.count + m1,
                                       c2.error + m1) < 0)
                    return -1;
            }
            else {
                i1 = spsv_int64_counter_new(T1, c2.item, c2.count + m1,
                                               c2.error + m1);
            }
            kh_val(T1->hashmap, iter) = i1;
        }
        else if (absent < 0) {
            PyErr_NoMemory();
            return -1;
        }
        i2 = T2->list[i2].next;
    }
    return 0;
}


CRICK_INLINE int spsv_int64_update_ndarray(spsv_int64_t *T,
                                              PyArrayObject *x,
                                              PyArrayObject *w) {
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext;
    PyArrayObject *op[2];
    npy_uint32 flags;
    npy_uint32 op_flags[2];
    PyArray_Descr *dtypes[2] = {NULL};

    npy_intp *innersizeptr, *strideptr;
    char **dataptr;

    npy_intp ret = -1;
    NPY_BEGIN_THREADS_DEF;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(x) == 0 || PyArray_SIZE(w) == 0) {
        return 0;
    }

    op[0] = x;
    op[1] = w;
    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED | NPY_ITER_REFS_OK;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;

    dtypes[0] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[0] == NULL) {
        goto finish;
    }
    Py_INCREF(dtypes[0]);
    dtypes[1] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[1] == NULL) {
        goto finish;
    }
    Py_INCREF(dtypes[1]);

    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto finish;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        goto finish;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    if (!NpyIter_IterationNeedsAPI(iter))
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));

    do {
        char *data_x = dataptr[0];
        char *data_w = dataptr[1];
        npy_intp stride_x = strideptr[0];
        npy_intp stride_w = strideptr[1];
        npy_intp count = *innersizeptr;

        while (count--) {
            spsv_int64_add(T, *(npy_int64 *)data_x,
                              *(npy_int64 *)data_w);

            data_x += stride_x;
            data_w += stride_w;
        }
    } while (iternext(iter));

    NPY_END_THREADS;

    ret = 0;

finish:
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            return -1;
        }
    }
    return ret;
}

typedef struct {
    PyObject* item;
    npy_int64 count;
    npy_int64 error;
} counter_object_t;


typedef struct {
    npy_intp next;
    npy_intp prev;
    counter_object_t counter;
} node_object_t;


typedef struct {
    SPSV_HEAD
    node_object_t *list;
    khash_t(object) *hashmap;
} spsv_object_t;


CRICK_INLINE spsv_object_t *spsv_object_new(int capacity) {
    spsv_object_t *T = (spsv_object_t *)malloc(sizeof(*T));
    if (T == NULL)
        goto fail;

    T->list = (node_object_t *)malloc(capacity * sizeof(node_object_t));
    if (T->list == NULL)
        goto fail;

    T->capacity = capacity;
    T->size = 0;
    T->head = NIL;
    T->hashmap = kh_init(object);

    return T;

fail:
    if (T->list != NULL) free(T->list);
    if (T != NULL) free(T);
    return NULL;
}


CRICK_INLINE void spsv_object_free(spsv_object_t *T) {
    khiter_t iter;
    for (iter = kh_begin(T->hashmap); iter != kh_end(T->hashmap); ++iter) {
        if (kh_exist(T->hashmap, iter))
            Py_DECREF(kh_key(T->hashmap, iter));
    }
    kh_destroy(object, T->hashmap);
    free(T->list);
    free(T);
}


CRICK_INLINE int counter_object_ge(counter_object_t c1,
                                     counter_object_t c2,
                                     npy_int64 offset) {
    npy_int64 count = c2.count + offset;
    npy_int64 error = c2.error + offset;
    return (c1.count > count || (c1.count == count && c1.error <= error));
}


CRICK_INLINE void spsv_object_counter_insert(spsv_object_t *T,
                                                npy_intp c, npy_intp prev) {
    npy_intp tail = T->list[T->head].prev;
    while(1) {
        if (counter_object_ge(T->list[prev].counter,
                                T->list[c].counter, 0))
            break;
        prev = T->list[prev].prev;
        if (prev == tail) {
            T->head = c;
            break;
        }
    }
    T->list[c].next = T->list[prev].next;
    T->list[c].prev = prev;
    T->list[T->list[prev].next].prev = c;
    T->list[prev].next = c;
}


CRICK_INLINE npy_intp spsv_object_counter_new(spsv_object_t *T,
                                                PyObject* item, npy_int64 count,
                                                npy_int64 error) {
    npy_intp c = T->size;
    T->size++;
    Py_INCREF(item);

    T->list[c].counter.item = item;
    T->list[c].counter.count = count;
    T->list[c].counter.error = error;

    if (T->head == NIL) {
        T->head = c;
        T->list[c].prev = c;
        T->list[c].next = c;
    }
    else {
        npy_intp tail = T->list[T->head].prev;
        spsv_object_counter_insert(T, c, tail);
    }
    return c;
}


CRICK_INLINE void spsv_object_rebalance(spsv_object_t *T, npy_intp index) {
    npy_intp prev;
    if (T->head == index) {
        /* Counts can only increase */
        return;
    }
    prev = T->list[index].prev;

    if (counter_object_ge(T->list[prev].counter,
                            T->list[index].counter, 0))
        return;

    /* Counter needs to be moved. Remove then insert. */
    T->list[T->list[index].next].prev = prev;
    T->list[prev].next = T->list[index].next;
    spsv_object_counter_insert(T, index, prev);
}


CRICK_INLINE int spsv_object_swap(spsv_object_t *T, npy_intp index,
                                    PyObject* item, npy_int64 count,
                                    npy_int64 error) {
    /* Remove the old item from the hashmap */
    PyObject* old_item = T->list[index].counter.item;
    khiter_t iter = kh_get(object, T->hashmap, old_item);
    if (iter == kh_end(T->hashmap)) return -1;

    if (PyErr_Occurred()) return -1;

    kh_del(object, T->hashmap, iter);

    Py_DECREF(old_item);
    Py_INCREF(item);

    T->list[index].counter.item = item;
    T->list[index].counter.error = error;
    T->list[index].counter.count = count;
    spsv_object_rebalance(T, index);
    return 0;
}


CRICK_INLINE int spsv_object_add(spsv_object_t *T, PyObject* item,
                                   npy_int64 count) {
    int absent;
    npy_intp index;

    /* Get the pointer to the bucket */
    khiter_t iter = kh_put(object, T->hashmap, item, &absent);
    /* If the key is an object, we need to check for hash failure */
    if (PyErr_Occurred()) return -1;
    if (absent > 0) {
        /* New item */
        if (T->size == T->capacity) {
            /* we're full, replace the min counter */
            index = T->list[T->head].prev;
            if (spsv_object_swap(T, index, item,
                                   T->list[index].counter.count + 1,
                                   T->list[index].counter.count))
                return -1;
        } else {
            /* Not full, allocate a new counter */
            index = spsv_object_counter_new(T, item, count, 0);
        }
        kh_val(T->hashmap, iter) = index;
    }
    else if (absent == 0) {
        /* The counter exists, just update it */
        index = kh_val(T->hashmap, iter);
        T->list[index].counter.count += count;
        spsv_object_rebalance(T, index);
    }
    else {
        PyErr_NoMemory();
        return -1;
    }
    return 1;
}


CRICK_INLINE int spsv_object_set_state(spsv_object_t *T,
                                         counter_object_t *counters,
                                         npy_intp size) {
    int i, absent;
    npy_intp index;
    if (size > T->capacity) {
        PyErr_SetString(PyExc_ValueError,
                        "deserialization failed, size > capacity");
        return -1;
    }
    for (i=0; i < size; i++) {
        counter_object_t c = counters[i];
        /* Get the pointer to the bucket */
        khiter_t iter = kh_put(object, T->hashmap, c.item, &absent);
        /* If the key is an object, we need to check for hash failure */
        if (PyErr_Occurred()) return -1;
        if (absent > 0) {
            index = spsv_object_counter_new(T, c.item, c.count, c.error);
            kh_val(T->hashmap, iter) = index;
        }
        else if (absent == 0) {
            PyErr_SetString(PyExc_ValueError,
                            "deserialization failed, duplicate items found");
            return -1;
        }
        else {
            PyErr_NoMemory();
            return -1;
        }
    }
    return 1;
}


CRICK_INLINE int spsv_object_merge(spsv_object_t *T1, spsv_object_t *T2) {
    npy_int64 m1, m2;
    npy_intp i1, i2;
    int i;
    /* Nothing to do */
    if (T2->size == 0) return 0;

    /* Get the minimum counts from each */
    if (T1->size < T1->capacity)
        m1 = 0;
    else
        m1 = T1->list[T1->list[T1->head].prev].counter.count;

    if (T2->size < T2->capacity)
        m2 = 0;
    else
        m2 = T2->list[T2->list[T2->head].prev].counter.count;

    /* Iterate through T1, updating it inplace */
    for (i1 = 0; i1 < T1->size; i1++) {
        khiter_t iter = kh_get(object, T2->hashmap, T1->list[i1].counter.item);

        if (PyErr_Occurred()) return -1;

        if (iter != kh_end(T2->hashmap)) {
            /* item is in both T1 and T2 */
            i2 = kh_val(T2->hashmap, iter);
            T1->list[i1].counter.count += T2->list[i2].counter.count;
            T1->list[i1].counter.error += T2->list[i2].counter.error;
        }
        else {
            /* item is in only T1 */
            T1->list[i1].counter.count += m2;
            T1->list[i1].counter.error += m2;
        }
        spsv_object_rebalance(T1, i1);
    }

    /* Iterate through T2, adding in any missing items */
    i2 = T2->head;
    for (i = 0; i < T2->size; i++) {
        int absent;
        counter_object_t c2 = T2->list[i2].counter;
        khiter_t iter = kh_put(object, T1->hashmap, c2.item, &absent);
        if (PyErr_Occurred()) return -1;

        if (absent > 0) {
            /* Item isn't in T1, maybe add it */
            if (T1->size == T1->capacity) {
                i1 = T1->list[T1->head].prev;

                /* If all counts in T1 are >= T2 + m1 then we're done here */
                if (counter_object_ge(T1->list[i1].counter, c2, m1)) break;

                if (spsv_object_swap(T1, i1, c2.item, c2.count + m1,
                                       c2.error + m1) < 0)
                    return -1;
            }
            else {
                i1 = spsv_object_counter_new(T1, c2.item, c2.count + m1,
                                               c2.error + m1);
            }
            kh_val(T1->hashmap, iter) = i1;
        }
        else if (absent < 0) {
            PyErr_NoMemory();
            return -1;
        }
        i2 = T2->list[i2].next;
    }
    return 0;
}


CRICK_INLINE int spsv_object_update_ndarray(spsv_object_t *T,
                                              PyArrayObject *x,
                                              PyArrayObject *w) {
    NpyIter *iter = NULL;
    NpyIter_IterNextFunc *iternext;
    PyArrayObject *op[2];
    npy_uint32 flags;
    npy_uint32 op_flags[2];
    PyArray_Descr *dtypes[2] = {NULL};

    npy_intp *innersizeptr, *strideptr;
    char **dataptr;

    npy_intp ret = -1;
    NPY_BEGIN_THREADS_DEF;

    /* Handle zero-sized arrays specially */
    if (PyArray_SIZE(x) == 0 || PyArray_SIZE(w) == 0) {
        return 0;
    }

    op[0] = x;
    op[1] = w;
    flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_BUFFERED | NPY_ITER_REFS_OK;
    op_flags[0] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;
    op_flags[1] = NPY_ITER_READONLY | NPY_ITER_ALIGNED;

    dtypes[0] = PyArray_DescrFromType(NPY_OBJECT);
    if (dtypes[0] == NULL) {
        goto finish;
    }
    Py_INCREF(dtypes[0]);
    dtypes[1] = PyArray_DescrFromType(NPY_INT64);
    if (dtypes[1] == NULL) {
        goto finish;
    }
    Py_INCREF(dtypes[1]);

    iter = NpyIter_MultiNew(2, op, flags, NPY_KEEPORDER, NPY_SAFE_CASTING,
                            op_flags, dtypes);
    if (iter == NULL) {
        goto finish;
    }

    iternext = NpyIter_GetIterNext(iter, NULL);
    if (iternext == NULL) {
        goto finish;
    }
    dataptr = NpyIter_GetDataPtrArray(iter);
    strideptr = NpyIter_GetInnerStrideArray(iter);
    innersizeptr = NpyIter_GetInnerLoopSizePtr(iter);

    if (!NpyIter_IterationNeedsAPI(iter))
        NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter));

    do {
        char *data_x = dataptr[0];
        char *data_w = dataptr[1];
        npy_intp stride_x = strideptr[0];
        npy_intp stride_w = strideptr[1];
        npy_intp count = *innersizeptr;

        while (count--) {
            spsv_object_add(T, *(PyObject* *)data_x,
                              *(npy_int64 *)data_w);

            data_x += stride_x;
            data_w += stride_w;
        }
    } while (iternext(iter));

    NPY_END_THREADS;

    ret = 0;

finish:
    Py_XDECREF(dtypes[0]);
    Py_XDECREF(dtypes[1]);
    if (iter != NULL) {
        if (NpyIter_Deallocate(iter) != NPY_SUCCEED) {
            return -1;
        }
    }
    return ret;
}
/*
 * END LOOP
 */


/* float64 definitions are just a thin wrapper around int64, viewing the bytes
 * as int64. Define a small helper to view float64 as int64: */

CRICK_INLINE npy_int64 asint64(npy_float64 key) {
  return *(npy_int64 *)(&key);
}
