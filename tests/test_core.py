
import rastereasy
import numpy as np


def test_all():
    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im)
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image_names=rastereasy.Geoimage(name_im,names=names)
    Image_names.crop(100,150,100,150,inplace=True)
    im1=Image.crop(100,150,100,150)
    im2=Image.crop(250,300,250,300)
    diff=im2-im1
    diff2=diff.abs()
    diff.abs(inplace=True)
    assert (diff-diff2).abs().sum() == 0

    Image.info()
    Image_names.get_meta()['dtype']
    row=20
    col=10
    pixel_row = 15
    pixel_col = 22
    assert np.all((Image[row,:][:,pixel_col]-Image[row,pixel_col]) == 0)
    assert np.all((Image[:,col][:,pixel_row]-Image[pixel_row,col]) == 0)


    Image_std,scaler_std=Image.standardize(dest_name='./tests/res_test/test_image_S2_std.tif')
    Image_minmax,scaler_minmax=Image.standardize(dest_name='./tests/res_test/test_image_S2_minmax.tif',type='minmax')


    Image=rastereasy.Geoimage(name_im)
    Image_apply_std=Image.standardize(scaler_std)
    Image_apply_minmax=Image.standardize(scaler_minmax)
    assert (np.sum(np.abs(Image_apply_std.image-Image_std.image))) ==0
    assert(np.sum(np.abs(Image_apply_minmax.image-Image_minmax.image)))==0


    Image_recovered_std=Image_std.inverse_standardize(scaler_std)
    Image_recovered_minmax=Image_minmax.inverse_standardize(scaler_minmax)

    Image=rastereasy.Geoimage(name_im)
    Image.standardize(inplace=True)
    assert((Image_std-Image).abs().sum())==0

    Image=rastereasy.Geoimage(name_im)

    Image.standardize(type='minmax',inplace=True)
    assert((Image_minmax-Image).abs().sum())==0

    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im)
    Image.standardize(scaler_std,
                      dest_name='./tests/res_test/test_image_S2_std2.tif',
                      inplace=True)
    assert((Image_std-Image).abs().sum())==0

    Image=rastereasy.Geoimage(name_im)
    Image.standardize(scaler_minmax,
                      dest_name='./tests/res_test/test_image_S2_minmax2.tif',
                      inplace=True)
    assert((Image_minmax-Image).abs().sum()) ==0

    name_im='./tests/res_test/test_image_S2_minmax2.tif'
    I1=rastereasy.Geoimage(name_im)
    name_im='./tests/res_test/test_image_S2_minmax.tif'
    I2=rastereasy.Geoimage(name_im)
    print((I1-I2).abs().sum())
    assert((I1-I2).abs().sum()) ==0

    name_im='./tests/res_test/test_image_S2_std.tif'
    I1=rastereasy.Geoimage(name_im)
    name_im='./tests/res_test/test_image_S2_std2.tif'
    I2=rastereasy.Geoimage(name_im)
    print((I1-I2).abs().sum())
    assert((I1-I2).abs().sum())==0


    name_im='./tests/res_test/test_image_S2_std.tif'
    Image=rastereasy.Geoimage(name_im)
    Image.inverse_standardize(scaler_std, inplace=True)
    assert(np.sum(np.abs(Image_recovered_std.image-Image.image)))==0

    name_im='./tests/res_test/test_image_S2_minmax.tif'
    Image=rastereasy.Geoimage(name_im)
    Image.inverse_standardize(scaler_minmax, inplace=True)
    assert(np.sum(np.abs(Image_recovered_minmax.image-Image.image)))==0



    name_im='./tests/data/sentinel.tif'
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image=rastereasy.Geoimage(name_im,names=names,history=True)
    Image.select_bands(["SWIR1","NIR","RE1"],reformat_names=False, inplace=True)
    Image=rastereasy.Geoimage(name_im)
    Image.select_bands(["8",10,8],reformat_names=True, inplace=True)
    Image.change_names({"WA":2,"NIR2":3,"NIR1":1})

    name_im='./tests/data/sentinel.tif'
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image=rastereasy.Geoimage(name_im,names=names,history=True)
    Image_select=Image.select_bands(["SWIR1","NIR","B","RE1"],dest_name='./tests/data/results/band_selection/select.tif')
    Image=rastereasy.Geoimage(name_im)
    Image_select=Image.select_bands(['8','8',10,10])
    im=Image_names.numpy_channel_first()
    im=Image_names.numpy_channel_last()
    bands=["1","4","3"]
    im=Image.numpy_channel_first(bands=bands)
    bands=[1,4,3,8]
    im=Image.numpy_channel_last(bands=bands)
    bands=["R","G","B"]
    im=Image_names.numpy_channel_first(bands=bands)
    bands=["R","NIR","G","B"]
    im=Image_names.numpy_channel_last(bands=bands)
    table=Image.numpy_table()
    image_recovered=rastereasy.table2image(table,Image.shape)
    assert(np.sum(np.abs(image_recovered-Image.numpy_channel_first())))==0
    image_recovered=rastereasy.table2image(table,Image.shape,channel_first=False)
    assert(np.sum(np.abs(image_recovered-Image.numpy_channel_last())))==0




    image_recovered=Image.image_from_table(table[:,1:4],names={"R":3,"G":2,"B":1},dest_name='./tests/data/results/change_bands/image_from_table1.tif')
    Image.activate_history()
    Image.upload_table(table[:,2:5],names={"R":3,"G":2,"B":1},dest_name='./tests/data/results/change_bands/image_from_table2.tif')
    bands=["R","NIR","G"]
    table=Image_names.numpy_table(bands)
    image_recovered=rastereasy.table2image(table,Image_names.shape)
    image_recovered=rastereasy.table2image(table,Image_names.shape,channel_first=False)

    bands=[3,"4",8]
    image_recovered=rastereasy.table2image(table,Image_names.shape)
    image_recovered=rastereasy.table2image(table,Image_names.shape,channel_first=False)
    image_recovered=Image_names.image_from_table(table)
    Image_names.info()
    Image_names.upload_table(table)
    Image_names.info()


    im1=rastereasy.Geoimage('./tests/data/RGB_common_1.tif')
    im2=rastereasy.Geoimage('./tests/data/RGB_common_2.tif')

    im1_common_res1, im2_common_res1 = rastereasy.extract_common_areas(im1, im2)
    im1_common_res2, im2_common_res2 = rastereasy.extract_common_areas(im1, im2,resolution='max')




    im1_extend_res1, im2_extend_res1 = rastereasy.extend_common_areas(im1, im2)
    im1_extend_res2, im2_extend_res2 = rastereasy.extend_common_areas(im1, im2,resolution='max')


    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im,history=True)


    # In[64]:




    # 1st option : we modify the image
    deb_row=50
    deb_col=100
    end_row = 200
    end_col=300

    Image.crop(deb_row,end_row,deb_col,end_col,
              inplace=True)


    Image_names=rastereasy.Geoimage(name_im,history=True,names=names)
    image_crop=Image_names.crop(deb_row,end_row,deb_col,end_col)


    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im,history=True)


    deb_row=50
    deb_col=100
    end_row = 200
    end_col=300
    lat1,lon1=Image.pixel2latlon(deb_row,deb_col)
    lat2,lon2=Image.pixel2latlon(end_row,end_col)
    est_deb_row,est_deb_col=Image.latlon2pixel(lat1,lon1)
    est_end_row,est_end_col=Image.latlon2pixel(lat2,lon2)
    Image.crop(lon1,lon2,lat1,lat2,
               pixel=False,
              inplace=True)


    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im)
    deb_row=50
    deb_col=100
    end_row = 200
    end_col=300
    lat1,lon1=Image.pixel2latlon(deb_row,deb_col)
    lat2,lon2=Image.pixel2latlon(end_row,end_col)
    est_deb_row,est_deb_col=Image.latlon2pixel(lat1,lon1)
    est_end_row,est_end_col=Image.latlon2pixel(lat2,lon2)

    Image=rastereasy.Geoimage(name_im)
    image_crop=Image.crop(lon1,lon2,lat1,lat2,pixel=False)
    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im,names=names)


    final_resolution=25
    # 1st option : we modify the image
    print('Before resampling')
    Image.resampling(final_resolution=final_resolution,
                     inplace=True)
    print('After resampling')


    Image=rastereasy.Geoimage(name_im,names=names)
    image_resampled=Image.resampling(final_resolution=final_resolution)


    name_im = './tests/data/sentinel.tif'
    image=rastereasy.Geoimage(name_im)



    image=rastereasy.Geoimage(name_im)
    image.reproject('EPSG:27700',inplace=True)
    image_reprojected=image.reproject('EPSG:3413')



    image_reprojected_bis=image_reprojected.reproject('EPSG:32630')


    row1 = 3
    col1 = 19
    row2 = 80
    col2 = 25
    lat1,lon1=Image.pixel2latlon(row1,col1)
    lat2,lon2=Image.pixel2latlon(row2,col2)
    est_row1,est_col1=Image.latlon2pixel(lat1,lon1)
    est_row2,est_col2=Image.latlon2pixel(lat2,lon2)

    im=rastereasy.Geoimage('./tests/data/sentinel.tif',history=True).select_bands(4)

    im.crop(100,200,100,200)
    im.resampling(40,inplace=True)
    names={"LAI":1}
    im.change_names(names)

    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im,history=True)
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image_names=rastereasy.Geoimage(name_im,names=names,history=True)

    Image.remove_bands(['8','4','10','1'],reformat_names=True,inplace=True)

    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im,history=True)
    #names = {"CO" : 1,"B": 2,"G":3,"R":4,"RE1":5,"RE2":6,"RE3":7,"NIR":8,"WA":9,"SWIR1":10,"SWIR2":11,"SWIR3":12}
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image=rastereasy.Geoimage(name_im,history=True)
    Image_names=rastereasy.Geoimage(name_im,names=names,history=True)
    Image_names_removed=Image_names.remove_bands(["NIR","SWIR1"])
    Image_names_removed.info()
    Image_removed=Image.remove_bands(["8","10"])
    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im)
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image_names=rastereasy.Geoimage(name_im,names=names)
    Image_rem=rastereasy.rasters.remove_bands(Image,[4,8,10])
    Image_name_rem=rastereasy.rasters.remove_bands(Image_names,["NIR","G","B"],reformat_names=False)

    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im,history=True)
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image_names=rastereasy.Geoimage(name_im,names=names,history=True)
    NIR=Image_names.select_bands('NIR')
    G=Image_names.select_bands('G')
    B=Image_names.select_bands('B')
    Image_names_removed=Image_names.remove_bands(["NIR","G","CO","SWIR2","B","R","RE1","SWIR1","SWIR3"])
    Image_names_removed.add_band(NIR.image,name_band="NIR", inplace=True)
    Image_names_removed.add_band(G.image,name_band="GR", inplace=True)
    Image_names_removed.add_band(B.image,name_band="B",dtype='int16', inplace=True)

    image=rastereasy.Geoimage(name_im,history=True)
    image2=image.resampling(20)
    image.select_bands('1').save('./tests/res_test/st1.tif')
    image2.select_bands('4').save('./tests/res_test/st2.tif')

    im1=rastereasy.Geoimage('./tests/res_test/st1.tif')
    im2=rastereasy.Geoimage('./tests/res_test/st2.tif')
    st=rastereasy.files2stack(['./tests/res_test/st1.tif','./tests/res_test/st2.tif'], resolution=20)

    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    im1=rastereasy.Geoimage('./tests/data/sentinel.tif',names=names,history=True)
    im1.kmeans(4,dest_name='./tests/res_test/kmeans.tif')
    im2=rastereasy.Geoimage('./tests/res_test/kmeans.tif',history=True)
    im2=im1.stack(im2)
    im2.stack(im2,inplace=True)



    # Read image
    im1=rastereasy.Geoimage(name_im)
    # Extract red and nir bands
    r=im1.select_bands(4)
    nir=im1.select_bands(8)

    # Computing ndvi
    ndvi=(nir-r)/(nir+r+1e-6)
    # Change the name of the band dor consistency
    ndvi.change_names({'ndvi':1})
    # plot infos

    threshold=0.4
    baresoil=ndvi<=threshold
    baresoil.info()

    ndvi.info()
    baresoil2=ndvi.where(ndvi<=threshold,1,0)
    baresoil2.info()
    (baresoil2==baresoil).sum()-baresoil.shape[0]*baresoil.shape[1]
    assert  ((baresoil2==baresoil).sum()-baresoil.shape[0]*baresoil.shape[1]) ==0

    baresoil3=ndvi*0
    baresoil3[ndvi<=threshold]=1
    baresoil3[ndvi>threshold]=0
    (baresoil3==baresoil2).sum()-baresoil3.shape[0]*baresoil3.shape[1]
    assert ((baresoil3==baresoil2).sum()-baresoil3.shape[0]*baresoil3.shape[1]) ==0

    im=rastereasy.Geoimage('./tests/data/sentinel.tif')



    im2=im.crop(20,200,150,300).resampling(30).remove_bands(['3',4,5,10])


    Image=rastereasy.Geoimage(name_im)

    classif_all_bands,kmeans_model=Image.kmeans(n_clusters=4,random_state=None,nb_points=None)

    classif_all_bands_nostd,_=Image.kmeans(n_clusters=10,random_state=2,standardization=False)

    classif_all_bands_std,_=Image.kmeans(n_clusters=10,random_state=2,bands=["8",3,1,2],standardization=False)

    classif_half1,model=Image.crop(0,333,0,150).kmeans(n_clusters=4,nb_points=None)

    classif_half1.info()

    classif_half2=Image.crop(0,333,150,333).apply_ML_model(model)
    classif_half2.info()
    classif1 ,classif2=rastereasy.extend_common_areas(classif_half1,classif_half2)
    # 4) merge images
    classif_all=classif1+classif2





    shapefile_path = './tests/shp/Roi_G5.shp'
    rastereasy.shpfiles.shp2raster(shapefile_path,'./tests/res_test/raster5m.tif',resolution=5)
    rastereasy.shpfiles.shp2raster(shapefile_path,'./tests/res_test/raster10m.tif',resolution=10)


    # ## 2) Import as Geoimage

    # In[150]:


    raster10m=rastereasy.Geoimage('./tests/res_test/raster10m.tif')
    raster5m=rastereasy.Geoimage('./tests/res_test/raster5m.tif')




    georaster5=rastereasy.shpfiles.shp2geoim(shapefile_path,resolution=5)
    georaster10=rastereasy.shpfiles.shp2geoim(shapefile_path,resolution=10)



    # In[156]:

    im1=rastereasy.files2stack('./tests/tostack/',ext='tif')

    keep_size=True
    value=3
    im=rastereasy.Geoimage('./tests/tostack/G5_B2.tif')
    im.stack(rastereasy.Geoimage('./tests/tostack/G5_B3.tif'),reformat_names=True)
    im.stack(rastereasy.Geoimage('./tests/tostack/G5_B4.tif'),reformat_names=True)
    im.stack(rastereasy.Geoimage('./tests/tostack/G5_B8.tif'),reformat_names=True)

    ime=im.extract_from_shapefile(shapefile_path,value,keep_size=keep_size)



    keep_size=False
    value=4


    im.extract_from_shapefile(shapefile_path,value,keep_size=keep_size,inplace=True)


    #Read a stack
    shapefile_path = './tests/shp/Roi_G5.shp'

    shp=rastereasy.shpfiles.shp2geoim(shapefile_path,resolution=10)


    # In[160]:


    keep_size=True
    value=3
    ime=im.extract_from_shapeimage(shp,value,keep_size=keep_size)

    shp=rastereasy.shpfiles.shp2geoim(shapefile_path,resolution=3)


    # In[163]:


    keep_size=False
    value=4

    #Read a stack
    name_im='./tests/data/sentinel.tif'
    Image=rastereasy.Geoimage(name_im)
    names = {"NIR":8,"G":3,"CO" : 1,"SWIR2":11,"B": 2,"R":4,"RE1":5,"RE2":6,"RE3":7,"WA":9,"SWIR1":10,"SWIR3":12}
    Image_names=rastereasy.Geoimage(name_im,names=names)
    Image_names.crop(0,300,0,333,inplace=True)

    raw = np.random.rand(300, 333, 3)

    mass_functions = raw / np.sum(raw, axis=2, keepdims=True)

    assert np.allclose(np.sum(mass_functions, axis=2), 1.0, atol=1e-6)
    # mass_functions est maintenant un tableau (1000, 1000, 3) avec des masses valides
    im1 = Image_names.upload_image(mass_functions, channel_first=False)
    raw = np.random.rand(300, 333, 3)
    mass_functions = raw / np.sum(raw, axis=2, keepdims=True)
    im2 = Image_names.upload_image(mass_functions, channel_first=False)
    raw = np.random.rand(300, 333, 3)
    mass_functions = raw / np.sum(raw, axis=2, keepdims=True)
    im3 = Image_names.upload_image(mass_functions, channel_first=False)
    raw = np.random.rand(300, 333, 3)
    mass_functions = raw / np.sum(raw, axis=2, keepdims=True)
    im4 = Image_names.upload_image(mass_functions, channel_first=False)
    fuse,confl=rastereasy.InferenceTools.fuse_dempster_shafer_2hypotheses(im1,im2,im3,im4)
    fuse.sum(axis='band')
    Image_names.select_bands('R',inplace=True)
    Image_names.info()


    fuse.sum(axis='band')
