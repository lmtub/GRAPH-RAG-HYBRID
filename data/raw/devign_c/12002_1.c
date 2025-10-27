static int virtio_ccw_hcall_notify(const uint64_t *args)

{

    uint64_t subch_id = args[0];

    uint64_t queue = args[1];

    SubchDev *sch;

    int cssid, ssid, schid, m;



    if (ioinst_disassemble_sch_ident(subch_id, &m, &cssid, &ssid, &schid)) {



    sch = css_find_subch(m, cssid, ssid, schid);

    if (!sch || !css_subch_visible(sch)) {






    virtio_queue_notify(virtio_ccw_get_vdev(sch), queue);

    return 0;


