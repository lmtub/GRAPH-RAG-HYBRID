void init_get_bits(GetBitContext *s,

                   UINT8 *buffer, int buffer_size)

{

    s->buffer= buffer;

    s->size= buffer_size;

    s->buffer_end= buffer + buffer_size;

#ifdef ALT_BITSTREAM_READER

    s->index=0;

#elif defined LIBMPEG2_BITSTREAM_READER

    s->buffer_ptr = buffer;

    s->bit_count = 16;

    s->cache = 0;

#elif defined A32_BITSTREAM_READER

    s->buffer_ptr = (uint32_t*)buffer;

    s->bit_count = 32;

    s->cache0 = 0;

    s->cache1 = 0;

#endif

    {

        OPEN_READER(re, s)

        UPDATE_CACHE(re, s)

//        UPDATE_CACHE(re, s)

        CLOSE_READER(re, s)

    }

#ifdef A32_BITSTREAM_READER

    s->cache1 = 0;

#endif

}
