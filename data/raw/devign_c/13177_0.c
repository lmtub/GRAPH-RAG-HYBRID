void tcg_set_frame(TCGContext *s, int reg, intptr_t start, intptr_t size)

{

    s->frame_start = start;

    s->frame_end = start + size;

    s->frame_reg = reg;

}
