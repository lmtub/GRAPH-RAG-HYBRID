static void set_gsi(KVMState *s, unsigned int gsi)

{

    assert(gsi < s->max_gsi);



    s->used_gsi_bitmap[gsi / 32] |= 1U << (gsi % 32);

}
