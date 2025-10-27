static void wav_capture_destroy (void *opaque)

{

    WAVState *wav = opaque;



    AUD_del_capture (wav->cap, wav);


}