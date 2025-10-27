static int vmsvga_post_load(void *opaque, int version_id)

{

    struct vmsvga_state_s *s = opaque;



    s->invalidated = 1;

    if (s->config)

        s->fifo = (uint32_t *) s->fifo_ptr;



    return 0;

}
