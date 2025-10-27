static uint16_t dummy_section(MemoryRegion *mr)

{

    MemoryRegionSection section = {

        .mr = mr,

        .offset_within_address_space = 0,

        .offset_within_region = 0,

        .size = int128_2_64(),

    };



    return phys_section_add(&section);

}
