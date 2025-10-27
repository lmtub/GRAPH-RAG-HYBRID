void bdrv_remove_aio_context_notifier(BlockDriverState *bs,

                                      void (*attached_aio_context)(AioContext *,

                                                                   void *),

                                      void (*detach_aio_context)(void *),

                                      void *opaque)

{

    BdrvAioNotifier *ban, *ban_next;



    QLIST_FOREACH_SAFE(ban, &bs->aio_notifiers, list, ban_next) {

        if (ban->attached_aio_context == attached_aio_context &&

            ban->detach_aio_context   == detach_aio_context   &&

            ban->opaque               == opaque)

        {

            QLIST_REMOVE(ban, list);

            g_free(ban);



            return;

        }

    }



    abort();

}
