bool virtio_is_big_endian(void)

{

#if defined(TARGET_WORDS_BIGENDIAN)

    return true;

#else

    return false;

#endif

}
