static int find_allocation(BlockDriverState *bs, off_t start,

                           off_t *data, off_t *hole)

{

    BDRVGlusterState *s = bs->opaque;

    off_t offs;



    if (!s->supports_seek_data) {

        return -ENOTSUP;

    }



    /*

     * SEEK_DATA cases:

     * D1. offs == start: start is in data

     * D2. offs > start: start is in a hole, next data at offs

     * D3. offs < 0, errno = ENXIO: either start is in a trailing hole

     *                              or start is beyond EOF

     *     If the latter happens, the file has been truncated behind

     *     our back since we opened it.  All bets are off then.

     *     Treating like a trailing hole is simplest.

     * D4. offs < 0, errno != ENXIO: we learned nothing

     */

    offs = glfs_lseek(s->fd, start, SEEK_DATA);

    if (offs < 0) {

        return -errno;          /* D3 or D4 */

    }

    assert(offs >= start);



    if (offs > start) {

        /* D2: in hole, next data at offs */

        *hole = start;

        *data = offs;

        return 0;

    }



    /* D1: in data, end not yet known */



    /*

     * SEEK_HOLE cases:

     * H1. offs == start: start is in a hole

     *     If this happens here, a hole has been dug behind our back

     *     since the previous lseek().

     * H2. offs > start: either start is in data, next hole at offs,

     *                   or start is in trailing hole, EOF at offs

     *     Linux treats trailing holes like any other hole: offs ==

     *     start.  Solaris seeks to EOF instead: offs > start (blech).

     *     If that happens here, a hole has been dug behind our back

     *     since the previous lseek().

     * H3. offs < 0, errno = ENXIO: start is beyond EOF

     *     If this happens, the file has been truncated behind our

     *     back since we opened it.  Treat it like a trailing hole.

     * H4. offs < 0, errno != ENXIO: we learned nothing

     *     Pretend we know nothing at all, i.e. "forget" about D1.

     */

    offs = glfs_lseek(s->fd, start, SEEK_HOLE);

    if (offs < 0) {

        return -errno;          /* D1 and (H3 or H4) */

    }

    assert(offs >= start);



    if (offs > start) {

        /*

         * D1 and H2: either in data, next hole at offs, or it was in

         * data but is now in a trailing hole.  In the latter case,

         * all bets are off.  Treating it as if it there was data all

         * the way to EOF is safe, so simply do that.

         */

        *data = start;

        *hole = offs;

        return 0;

    }



    /* D1 and H1 */

    return -EBUSY;

}
