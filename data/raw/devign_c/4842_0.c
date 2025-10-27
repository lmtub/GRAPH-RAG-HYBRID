uint32_t wm8750_adc_dat(void *opaque)

{

    WM8750State *s = (WM8750State *) opaque;

    uint32_t *data;



    if (s->idx_in >= sizeof(s->data_in))

        wm8750_in_load(s);



    data = (uint32_t *) &s->data_in[s->idx_in];

    s->req_in -= 4;

    s->idx_in += 4;

    return *data;

}
