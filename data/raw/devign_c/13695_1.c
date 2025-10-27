static int realloc_refcount_array(BDRVQcowState *s, uint16_t **array,

                                  int64_t *size, int64_t new_size)

{

    size_t old_byte_size, new_byte_size;

    uint16_t *new_ptr;



    /* Round to clusters so the array can be directly written to disk */

    old_byte_size = size_to_clusters(s, refcount_array_byte_size(s, *size))

                    * s->cluster_size;

    new_byte_size = size_to_clusters(s, refcount_array_byte_size(s, new_size))

                    * s->cluster_size;



    if (new_byte_size == old_byte_size) {

        *size = new_size;

        return 0;

    }



    assert(new_byte_size > 0);



    new_ptr = g_try_realloc(*array, new_byte_size);

    if (!new_ptr) {

        return -ENOMEM;

    }



    if (new_byte_size > old_byte_size) {

        memset((void *)((uintptr_t)new_ptr + old_byte_size), 0,

               new_byte_size - old_byte_size);

    }



    *array = new_ptr;

    *size  = new_size;



    return 0;

}
