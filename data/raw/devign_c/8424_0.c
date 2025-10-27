void *av_realloc_array(void *ptr, size_t nmemb, size_t size)

{

    if (size <= 0 || nmemb >= INT_MAX / size)

        return NULL;

    return av_realloc(ptr, nmemb * size);

}
