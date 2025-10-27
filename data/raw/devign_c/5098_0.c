static inline void* array_get_next(array_t* array) {

    unsigned int next = array->next;

    void* result;



    if (array_ensure_allocated(array, next) < 0)

	return NULL;



    array->next = next + 1;

    result = array_get(array, next);



    return result;

}
