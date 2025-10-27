unsigned int avpriv_toupper4(unsigned int x)

{

    return av_toupper(x & 0xFF) +

          (av_toupper((x >>  8) & 0xFF) << 8)  +

          (av_toupper((x >> 16) & 0xFF) << 16) +

          (av_toupper((x >> 24) & 0xFF) << 24);

}
