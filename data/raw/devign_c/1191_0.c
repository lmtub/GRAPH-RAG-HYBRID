static hwaddr vfio_container_granularity(VFIOContainer *container)

{

    return (hwaddr)1 << ctz64(container->iova_pgsizes);

}
