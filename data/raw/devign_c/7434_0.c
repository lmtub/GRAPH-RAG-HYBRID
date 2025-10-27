static void add_entry1(TiffEncoderContext *s,

                       enum TiffTags tag, enum TiffTypes type, int val)

{

    uint16_t w  = val;

    uint32_t dw = val;

    add_entry(s, tag, type, 1, type == TIFF_SHORT ? (void *)&w : (void *)&dw);

}
