static void temp_allocate_frame(TCGContext *s, int temp)

{

    TCGTemp *ts;

    ts = &s->temps[temp];

    s->current_frame_offset = (s->current_frame_offset + sizeof(tcg_target_long) - 1) & ~(sizeof(tcg_target_long) - 1);

    if (s->current_frame_offset + sizeof(tcg_target_long) > s->frame_end)

        tcg_abort();

    ts->mem_offset = s->current_frame_offset;

    ts->mem_reg = s->frame_reg;

    ts->mem_allocated = 1;

    s->current_frame_offset += sizeof(tcg_target_long);

}
