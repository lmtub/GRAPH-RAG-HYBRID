static void mark_request_serialising(BdrvTrackedRequest *req, uint64_t align)

{

    int64_t overlap_offset = req->offset & ~(align - 1);

    unsigned int overlap_bytes = ROUND_UP(req->offset + req->bytes, align)

                               - overlap_offset;



    if (!req->serialising) {

        req->bs->serialising_in_flight++;

        req->serialising = true;

    }



    req->overlap_offset = MIN(req->overlap_offset, overlap_offset);

    req->overlap_bytes = MAX(req->overlap_bytes, overlap_bytes);

}
