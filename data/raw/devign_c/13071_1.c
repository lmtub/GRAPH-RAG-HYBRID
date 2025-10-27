static void tracked_request_begin(BdrvTrackedRequest *req,
                                  BlockDriverState *bs,
                                  int64_t sector_num,
                                  int nb_sectors, bool is_write)
    *req = (BdrvTrackedRequest){
        .bs = bs,
        .sector_num = sector_num,
        .nb_sectors = nb_sectors,
        .is_write = is_write,
    };
    QLIST_INSERT_HEAD(&bs->tracked_requests, req, list);