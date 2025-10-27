static double tget_double(GetByteContext *gb, int le)

{

    av_alias64 i = { .u64 = le ? bytestream2_get_le64(gb) : bytestream2_get_be64(gb)};

    return i.f64;

}
