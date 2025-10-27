ParallelState *parallel_init(int index, CharDriverState *chr)

{

    ISADevice *dev;



    dev = isa_create("isa-parallel");

    qdev_prop_set_uint32(&dev->qdev, "index", index);

    qdev_prop_set_chr(&dev->qdev, "chardev", chr);

    if (qdev_init(&dev->qdev) < 0)

        return NULL;

    return &DO_UPCAST(ISAParallelState, dev, dev)->state;

}
