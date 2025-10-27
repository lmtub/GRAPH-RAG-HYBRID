av_cold void ff_schro_queue_init(FFSchroQueue *queue)

{

    queue->p_head = queue->p_tail = NULL;

    queue->size = 0;

}
