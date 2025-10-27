void *ff_png_zalloc(void *opaque, unsigned int items, unsigned int size)

{

    if(items >= UINT_MAX / size)

        return NULL;

    return av_malloc(items * size);

}
