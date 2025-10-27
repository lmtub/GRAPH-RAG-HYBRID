static void vfio_listener_release(VFIOContainer *container)

{

    memory_listener_unregister(&container->iommu_data.listener);

}
