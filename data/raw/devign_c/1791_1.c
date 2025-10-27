int ff_schro_queue_push_back(FFSchroQueue *queue, void *p_data)

{

    FFSchroQueueElement *p_new = av_mallocz(sizeof(FFSchroQueueElement));



    if (!p_new)

        return -1;



    p_new->data = p_data;



    if (!queue->p_head)

        queue->p_head = p_new;

    else

        queue->p_tail->next = p_new;

    queue->p_tail = p_new;



    ++queue->size;

    return 0;

}
