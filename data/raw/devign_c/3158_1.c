static unsigned tget(const uint8_t **p, int type, int le)

{

    switch (type) {

    case TIFF_BYTE:

        return *(*p)++;

    case TIFF_SHORT:

        return tget_short(p, le);

    case TIFF_LONG:

        return tget_long(p, le);

    default:

        return UINT_MAX;

    }

}
