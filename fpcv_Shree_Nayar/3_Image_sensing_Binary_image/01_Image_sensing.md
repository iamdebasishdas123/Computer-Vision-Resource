Now that we know how to capture an optical image,
0:05
we want to convert this optical image into a digital image
0:08
so that we can provide it to a computer vision
0:11
system for further analysis.
0:14
So in this lecture, we're going to start with a brief history
0:17
of imaging.
0:18
We look at the major inventions that led to the camera
0:21
as it stands today.
0:23
And we'll argue that perhaps the most important invention that's
0:26
taken place in the history of imaging
0:28
is that of the image sensor.
0:30
And in particular, we're going to look
0:32
at two types of image sensors--
0:34
CCD sensors and CMOS sensors.
0:38
And we'll look at how they work.
0:39
It really is a remarkable piece of technology.
0:44
And then, we'll look at the characteristics
0:46
of image sensors.
0:47
In particular resolution, which is the number of pixels
0:51
you end up with in an image.
0:52
Noise-- what are the different sources
0:55
of noise, undesirable effects that you end up with in images.
0:59
And then dynamic range, which is the range of brightness values
1:03
that an image sensor is able to measure.
1:06
And next, we'll talk about color.
1:09
Color is a really fascinating topic.
1:11
It turns out that color is really
1:13
human response to wavelength.
1:16
So we'll look at how image sensors can
1:18
be designed to capture color.
1:22
And next, we're going to look at the camera response function.
1:26
So if you have a point in the scene of a certain brightness
1:28
and it produces a brightness in the image,
1:31
if you increase the brightness of the point in the scene,
1:34
you can expect that the--
1:36
its image brightness is also going to increase.
1:39
But not necessarily linearly-- it
1:42
could be a nonlinear mapping between scene brightness
1:45
and image brightness.
1:46
And that's defined by the response function.
1:49
So we look at the camera response function
1:52
and how we can find the response function, given any camera.
1:57
And then we'll talk about the popular topic of HDR,
2:01
or high dynamic range imaging.
2:04
How can one use an image sensor with limited dynamic range
2:08
to capture images with wider dynamic range?
2:13
And finally, we'll look at some of nature's image sensors.
2:16
We'll look at some very interesting techniques
2:19
that nature has evolved, in terms of scanning images,
2:23
and finally, we'll talk about the retina of the human eye.