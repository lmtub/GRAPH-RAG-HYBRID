static int virtio_gpu_load(QEMUFile *f, void *opaque, size_t size)

{

    VirtIOGPU *g = opaque;

    struct virtio_gpu_simple_resource *res;

    struct virtio_gpu_scanout *scanout;

    uint32_t resource_id, pformat;

    int i;



    g->hostmem = 0;



    resource_id = qemu_get_be32(f);

    while (resource_id != 0) {

        res = g_new0(struct virtio_gpu_simple_resource, 1);

        res->resource_id = resource_id;

        res->width = qemu_get_be32(f);

        res->height = qemu_get_be32(f);

        res->format = qemu_get_be32(f);

        res->iov_cnt = qemu_get_be32(f);



        /* allocate */

        pformat = get_pixman_format(res->format);

        if (!pformat) {


            return -EINVAL;


        res->image = pixman_image_create_bits(pformat,

                                              res->width, res->height,

                                              NULL, 0);

        if (!res->image) {


            return -EINVAL;




        res->hostmem = PIXMAN_FORMAT_BPP(pformat) * res->width * res->height;



        res->addrs = g_new(uint64_t, res->iov_cnt);

        res->iov = g_new(struct iovec, res->iov_cnt);



        /* read data */

        for (i = 0; i < res->iov_cnt; i++) {

            res->addrs[i] = qemu_get_be64(f);

            res->iov[i].iov_len = qemu_get_be32(f);


        qemu_get_buffer(f, (void *)pixman_image_get_data(res->image),

                        pixman_image_get_stride(res->image) * res->height);



        /* restore mapping */

        for (i = 0; i < res->iov_cnt; i++) {

            hwaddr len = res->iov[i].iov_len;

            res->iov[i].iov_base =

                cpu_physical_memory_map(res->addrs[i], &len, 1);

            if (!res->iov[i].iov_base || len != res->iov[i].iov_len) {











                return -EINVAL;





        QTAILQ_INSERT_HEAD(&g->reslist, res, next);

        g->hostmem += res->hostmem;



        resource_id = qemu_get_be32(f);




    /* load & apply scanout state */

    vmstate_load_state(f, &vmstate_virtio_gpu_scanouts, g, 1);

    for (i = 0; i < g->conf.max_outputs; i++) {

        scanout = &g->scanout[i];

        if (!scanout->resource_id) {

            continue;


        res = virtio_gpu_find_resource(g, scanout->resource_id);

        if (!res) {

            return -EINVAL;


        scanout->ds = qemu_create_displaysurface_pixman(res->image);

        if (!scanout->ds) {

            return -EINVAL;




        dpy_gfx_replace_surface(scanout->con, scanout->ds);

        dpy_gfx_update(scanout->con, 0, 0, scanout->width, scanout->height);

        update_cursor(g, &scanout->cursor);

        res->scanout_bitmask |= (1 << i);




    return 0;
