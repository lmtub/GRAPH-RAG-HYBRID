static int add_metadata(int count, int type,

                        const char *name, const char *sep, TiffContext *s)

{

    switch(type) {

    case TIFF_DOUBLE: return add_doubles_metadata(count, name, sep, s);

    case TIFF_SHORT : return add_shorts_metadata(count, name, sep, s);

    case TIFF_STRING: return add_string_metadata(count, name, s);

    default         : return AVERROR_INVALIDDATA;

    };

}
