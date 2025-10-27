static void wm8750_audio_out_cb(void *opaque, int free_b)

{

    struct wm8750_s *s = (struct wm8750_s *) opaque;

    wm8750_out_flush(s);



    s->req_out = free_b;

    s->data_req(s->opaque, free_b >> 2, s->req_in >> 2);

}
