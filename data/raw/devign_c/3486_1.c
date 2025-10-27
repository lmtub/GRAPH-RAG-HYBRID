static av_cold int mov_text_encode_init(AVCodecContext *avctx)

{

    /*

     * For now, we'll use a fixed default style. When we add styling

     * support, this will be generated from the ASS style.

     */

    static const uint8_t text_sample_entry[] = {

        0x00, 0x00, 0x00, 0x00, // uint32_t displayFlags

        0x01,                   // int8_t horizontal-justification

        0xFF,                   // int8_t vertical-justification

        0x00, 0x00, 0x00, 0x00, // uint8_t background-color-rgba[4]

        // BoxRecord {

        0x00, 0x00,             // int16_t top

        0x00, 0x00,             // int16_t left

        0x00, 0x00,             // int16_t bottom

        0x00, 0x00,             // int16_t right

        // };

        // StyleRecord {

        0x00, 0x00,             // uint16_t startChar

        0x00, 0x00,             // uint16_t endChar

        0x00, 0x01,             // uint16_t font-ID

        0x00,                   // uint8_t face-style-flags

        0x12,                   // uint8_t font-size

        0xFF, 0xFF, 0xFF, 0xFF, // uint8_t text-color-rgba[4]

        // };

        // FontTableBox {

        0x00, 0x00, 0x00, 0x12, // uint32_t size

        'f', 't', 'a', 'b',     // uint8_t name[4]

        0x00, 0x01,             // uint16_t entry-count

        // FontRecord {

        0x00, 0x01,             // uint16_t font-ID

        0x05,                   // uint8_t font-name-length

        'S', 'e', 'r', 'i', 'f',// uint8_t font[font-name-length]

        // };

        // };

    };



    MovTextContext *s = avctx->priv_data;




    avctx->extradata_size = sizeof text_sample_entry;

    avctx->extradata = av_mallocz(avctx->extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);

    if (!avctx->extradata)

        return AVERROR(ENOMEM);



    av_bprint_init(&s->buffer, 0, AV_BPRINT_SIZE_UNLIMITED);



    memcpy(avctx->extradata, text_sample_entry, avctx->extradata_size);



    s->ass_ctx = ff_ass_split(avctx->subtitle_header);

    return s->ass_ctx ? 0 : AVERROR_INVALIDDATA;

}