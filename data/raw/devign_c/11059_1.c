static void cabac_init_decoder(HEVCContext *s)

{

    GetBitContext *gb = &s->HEVClc->gb;

    skip_bits(gb, 1);

    align_get_bits(gb);

    ff_init_cabac_decoder(&s->HEVClc->cc,

                          gb->buffer + get_bits_count(gb) / 8,

                          (get_bits_left(gb) + 7) / 8);

}
