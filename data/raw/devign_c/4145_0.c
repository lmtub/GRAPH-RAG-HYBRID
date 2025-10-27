static unsigned char get_ref_idx(AVFrame *frame)

{

    FrameDecodeData *fdd;

    NVDECFrame *cf;



    if (!frame || !frame->private_ref)

        return 255;



    fdd = (FrameDecodeData*)frame->private_ref->data;

    cf  = (NVDECFrame*)fdd->hwaccel_priv;



    return cf->idx;

}
