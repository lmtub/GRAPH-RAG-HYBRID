static void raw_aio_remove(RawAIOCB *acb)

{

    RawAIOCB **pacb;



    /* remove the callback from the queue */

    pacb = &posix_aio_state->first_aio;

    for(;;) {

        if (*pacb == NULL) {

            fprintf(stderr, "raw_aio_remove: aio request not found!\n");

            break;

        } else if (*pacb == acb) {

            *pacb = acb->next;

            qemu_aio_release(acb);

            break;

        }

        pacb = &(*pacb)->next;

    }

}
