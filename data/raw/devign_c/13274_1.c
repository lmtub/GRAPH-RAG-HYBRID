static void put_audio_specific_config(AVCodecContext *avctx)

{

    PutBitContext pb;

    AACEncContext *s = avctx->priv_data;



    init_put_bits(&pb, avctx->extradata, avctx->extradata_size*8);

    put_bits(&pb, 5, 2); //object type - AAC-LC

    put_bits(&pb, 4, s->samplerate_index); //sample rate index

    put_bits(&pb, 4, s->channels);

    //GASpecificConfig

    put_bits(&pb, 1, 0); //frame length - 1024 samples

    put_bits(&pb, 1, 0); //does not depend on core coder

    put_bits(&pb, 1, 0); //is not extension



    //Explicitly Mark SBR absent

    put_bits(&pb, 11, 0x2b7); //sync extension

    put_bits(&pb, 5,  AOT_SBR);

    put_bits(&pb, 1,  0);

    flush_put_bits(&pb);

}
