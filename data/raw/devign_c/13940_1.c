static unsigned int virtqueue_get_head(VirtQueue *vq, unsigned int idx)

{

    unsigned int head;



    /* Grab the next descriptor number they're advertising, and increment

     * the index we've seen. */

    head = vring_avail_ring(vq, idx % vq->vring.num);



    /* If their number is silly, that's a fatal mistake. */

    if (head >= vq->vring.num) {

        error_report("Guest says index %u is available", head);

        exit(1);

    }



    return head;

}
