static int iv_decode_frame(AVCodecContext *avctx,

                           const uint8_t *buf, int buf_size)

{

    Indeo3DecodeContext *s = avctx->priv_data;

    unsigned int image_width, image_height,

                 chroma_width, chroma_height;

    unsigned long flags, cb_offset, data_size,

                  y_offset, v_offset, u_offset, mc_vector_count;

    const uint8_t *hdr_pos, *buf_pos;



    buf_pos = buf;

    buf_pos += 18; /* skip OS header (16 bytes) and version number */



    flags = bytestream_get_le16(&buf_pos);

    data_size = bytestream_get_le32(&buf_pos);

    cb_offset = *buf_pos++;

    buf_pos += 3; /* skip reserved byte and checksum */

    image_height = bytestream_get_le16(&buf_pos);

    image_width  = bytestream_get_le16(&buf_pos);



    if(avcodec_check_dimensions(avctx, image_width, image_height))

        return -1;














    chroma_height = ((image_height >> 2) + 3) & 0x7ffc;

    chroma_width = ((image_width >> 2) + 3) & 0x7ffc;

    y_offset = bytestream_get_le32(&buf_pos);

    v_offset = bytestream_get_le32(&buf_pos);

    u_offset = bytestream_get_le32(&buf_pos);

    buf_pos += 4; /* reserved */

    hdr_pos = buf_pos;

    if(data_size == 0x80) return 4;



    if(FFMAX3(y_offset, v_offset, u_offset) >= buf_size-16) {

        av_log(s->avctx, AV_LOG_ERROR, "y/u/v offset outside buffer\n");

        return -1;




    if(flags & 0x200) {

        s->cur_frame = s->iv_frame + 1;

        s->ref_frame = s->iv_frame;

    } else {

        s->cur_frame = s->iv_frame;

        s->ref_frame = s->iv_frame + 1;




    buf_pos = buf + 16 + y_offset;

    mc_vector_count = bytestream_get_le32(&buf_pos);

    if(2LL*mc_vector_count >= buf_size-16-y_offset) {

        av_log(s->avctx, AV_LOG_ERROR, "mc_vector_count too large\n");

        return -1;




    iv_Decode_Chunk(s, s->cur_frame->Ybuf, s->ref_frame->Ybuf, image_width,

                    image_height, buf_pos + mc_vector_count * 2, cb_offset, hdr_pos, buf_pos,

                    FFMIN(image_width, 160));



    if (!(s->avctx->flags & CODEC_FLAG_GRAY))

    {



        buf_pos = buf + 16 + v_offset;

        mc_vector_count = bytestream_get_le32(&buf_pos);

        if(2LL*mc_vector_count >= buf_size-16-v_offset) {

            av_log(s->avctx, AV_LOG_ERROR, "mc_vector_count too large\n");

            return -1;




        iv_Decode_Chunk(s, s->cur_frame->Vbuf, s->ref_frame->Vbuf, chroma_width,

                chroma_height, buf_pos + mc_vector_count * 2, cb_offset, hdr_pos, buf_pos,

                FFMIN(chroma_width, 40));



        buf_pos = buf + 16 + u_offset;

        mc_vector_count = bytestream_get_le32(&buf_pos);

        if(2LL*mc_vector_count >= buf_size-16-u_offset) {

            av_log(s->avctx, AV_LOG_ERROR, "mc_vector_count too large\n");

            return -1;




        iv_Decode_Chunk(s, s->cur_frame->Ubuf, s->ref_frame->Ubuf, chroma_width,

                chroma_height, buf_pos + mc_vector_count * 2, cb_offset, hdr_pos, buf_pos,

                FFMIN(chroma_width, 40));






    return 8;
