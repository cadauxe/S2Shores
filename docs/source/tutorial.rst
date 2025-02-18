.. _tutorial:

========
Tutorial
========




....

### Python API

estimator = bathy_launcher.add_product(product_name, product_path, product_cls,
                                                        output_path, wave_params, nb_subtiles)

# Set the gravity provider.
estimator.set_gravity_provider(provider_info=gravity_type)


# Set the distoshore provider
estimator.set_distoshore_provider(provider_info=distoshore_file_path)

# Set the delta time provider.
estimator.set_delta_time_provider(provider_info=delta_times_path)

# Set the Roi provider.
estimator.set_roi_provider(provider_info=roi_file_path, limit_to_roi=limit_to_roi)

# Create subtiles (mandatory for setting debug area)
estimator.create_subtiles()


Creating plots



