The data in these training samples was created using the following:

Data was generated using Chrono::Sensors - Software developed by the Simulation Based Engineering Lab at UW-Madison
-lidar was placed at 0,0,0
-lidar has horizontal and vertical fields of view of 45° centered along the -x axis
-lidar uses 50x50 samples within the above FOV
-a single unit cube or unit sphere was placed with uniform randomness at x = (-4,-2), y = (-1,1), z = (-1,1)
-a solid wall is behind the object with face at x = -4.5

Data is in the following format:

Label: <label>
Position: <x>, <y>, <z>
Orientation (Quaternion): <e0>, <e1>, <e2>, <e3>
<pt x>, <pt y>, <pt z> 
<pt x>, <pt y>, <pt z> 
<pt x>, <pt y>, <pt z> 
<pt x>, <pt y>, <pt z> 
<pt x>, <pt y>, <pt z> 
<pt x>, <pt y>, <pt z> 
<pt x>, <pt y>, <pt z> 
...
