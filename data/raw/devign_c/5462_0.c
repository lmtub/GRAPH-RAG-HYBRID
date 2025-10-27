void bdrv_add_before_write_notifier(BlockDriverState *bs,

                                    NotifierWithReturn *notifier)

{

    notifier_with_return_list_add(&bs->before_write_notifiers, notifier);

}
