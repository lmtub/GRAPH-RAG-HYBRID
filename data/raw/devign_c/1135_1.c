void *grow_array(void *array, int elem_size, int *size, int new_size)

{

    if (new_size >= INT_MAX / elem_size) {

        av_log(NULL, AV_LOG_ERROR, "Array too big.\n");

        exit(1);

    }

    if (*size < new_size) {

        uint8_t *tmp = av_realloc(array, new_size*elem_size);

        if (!tmp) {

            av_log(NULL, AV_LOG_ERROR, "Could not alloc buffer.\n");

            exit(1);

        }

        memset(tmp + *size*elem_size, 0, (new_size-*size) * elem_size);

        *size = new_size;

        return tmp;

    }

    return array;

}
