void *ff_schro_queue_pop(FFSchroQueue *queue)

{

    FFSchroQueueElement *top = queue->p_head;



    if (top) {

        void *data = top->data;

        queue->p_head = queue->p_head->next;

        --queue->size;

        av_freep(&top);

        return data;

    }



    return NULL;

}
