void av_free(void *ptr)

{

#if CONFIG_MEMALIGN_HACK

    if (ptr)

        free((char *)ptr - ((char *)ptr)[-1]);

#elif HAVE_ALIGNED_MALLOC

    _aligned_free(ptr);

#else

    free(ptr);

#endif

}
