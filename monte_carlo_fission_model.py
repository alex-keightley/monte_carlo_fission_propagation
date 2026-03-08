###############################################################################################################################################
# Library Imports

import numpy as np
import matplotlib.pyplot as plt
from numpy import random as rdm
import time
from IPython.display import display, Latex
from matplotlib.colors import ListedColormap
from matplotlib import colormaps


###############################################################################################################################################
# Class Definition


# Class Uranium235_Fission_Model
# The class Uranium235_Fission_Model constructs a model that simulates Uranium fission reaction using Monte Carlo methods.
# The class has a class dictionary, several class methods for running the simulation and plotting results, and for running
# a trial of simulations. Details regarding each of these, as well as expected input arguments for creating class instances
# is detailed below in the appropriate section.
class Uranium235_Fission_Model:

    # Class Dictionary - shapes
    # The shapes dictionary associates a shape_type key with the shapes "dimensions", an array of values representing the dimensions of each
    # shape, as well as a "mask", a string for the name of the masking function that creates array to remove neutrons that leave the sample
    #     "sphere"   - has "dimensions" radius. One of two shapes that may be modeled using dimension- or volume-based construction.
    #     "cylinder" - has "dimensions" radius and height.
    #     "cube"     - has "dimensions" length. One of two shapes that may be modeled using dimension- or volume-based construction.
    #     "prism"    - has "dimensions" length, width, and height.
    shapes = {
        "sphere": {"dimensions": ["radius"], "mask": "sphere_mask"},
        "cylinder": {"dimensions": ["radius", "height"], "mask": "cylinder_mask"},
        "cube": {"dimensions": ["length"], "mask": "cube_mask"},
        "prism": {"dimensions": ["length", "width", "height"], "mask": "prism_mask"}
    }

    # Class Method - sphere_mask
    # The class method sphere_mask takes an input argument of an n x 3 dimensional array and returns an n-dimensional boolean array 
    # that is true wherever
    #     sqrt(x^2 + y^2 + z^2) <= r
    # This method is called within the class method captured to help determine when neutrons escape the sample.
    def sphere_mask(self, r):
        return (r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2 <= self.dimensions["radius"]**2)

    # Class Method - cube_mask
    # The class method sphere_mask takes an input argument of an n x 3 dimensional array and returns an n-dimensional boolean array 
    # that is true wherever for cube length L we have
    #     0 < x < l  AND  0 < y < L  AND  0 < z < L
    # This method is called within the class method captured to help determine when neutrons escape the sample.
    def cube_mask(self, r):
        return (r[:, 0] >= 0) & (r[:, 0] < self.dimensions["length"]) & \
               (r[:, 1] >= 0) & (r[:, 1] < self.dimensions["length"]) & \
               (r[:, 2] >= 0) & (r[:, 2] < self.dimensions["length"])

    # Class Method - cylinder_mask
    # The class method sphere_mask takes an input argument of an n x 3 dimensional array and returns an n-dimensional boolean array 
    # that is true wherever for cylinder height h and cylinder radius r we have
    #     0 < z < h  AND  sqrt(x^2 + y^2) <= r
    # This method is called within the class method captured to help determine when neutrons escape the sample.
    def cylinder_mask(self, r):
        return (r[:, 0]**2 + r[:, 1]**2 <= self.dimensions["radius"]**2) & \
               (r[:, 2] >= 0) & (r[:, 2] < self.dimensions["height"])

    # Class Method - prism_mask
    # The class method sphere_mask takes an input argument of an n x 3 dimensional array and returns an n-dimensional boolean array 
    # that is true wherever for prism length L, width w, and height h we have
    #     0 < x < L  AND  0 < y < w  AND 0 < z < h
    # This method is called within the class method captured to help determine when neutrons escape the sample.
    def prism_mask(self, r):
        return (r[:, 0] >= 0) & (r[:, 0] < self.dimensions["length"]) & \
               (r[:, 1] >= 0) & (r[:, 1] < self.dimensions["width"]) & \
               (r[:, 2] >= 0) & (r[:, 2] < self.dimensions["height"])


    # Class Method - __init__
    # The class initiation method, called whenever we create an instance of the Uranium235_Fission_Model, requires eight input arguments and
    # accepts up to 9, with the final input argument being a boolean flag for debug statements with a default value of false. The following
    # input arguments are expected upon construction:
    #
    #     N0                   - the number of of thermal neutrons to begin the simulation with in the zeroth generation
    #     mtrials              - the number of simulations to be completed during this trial
    #     shape_type           - a string input for the sample shape to build the model on. Currently implemented shapes are: "cube", "sphere", 
    #                                "cylinder", and "prism", where the arguments are case-sensitive.
    #     shape_parameters     - expected argument is a dictionary for the input shape parameters. Dictionary is expected in one of two forms:
    #                                (1) {"dimensions": [" ... ": value , "... ": value ]} where the dimensions listed may be "radius", "length", 
    #                                                      "width", or "height" as required by the shape and with their respective values, or
    #                                (2) {"volume": value} for regular shapes of one parameter from which the parameter can be solved from the volume.
    #                                                      Currently implemented shapes are "sphere", and "cube".
    #     purity               - the purity of the sample from 0 to 1; directly corresponds to the probability that an absorption event would 
    #                              cause a subsequent fission event. A purity of 1 guarantees fission from capture and a purity of 0 guarantees 
    #                              no fission events.
    #     neutron_multiplicity - the average number of thermal neutrons produced per fission event. For U-235 the average number of neutrons produced
    #                               per fission event is 2.4355.
    #     mean_free_Path       - The average distance (lambda) that a neutron will travel before being captured by a nucleus, so long as the thermal 
    #                               neutron remains inside the Uranium sample
    #     max_generation       - The maximum number of generations for fission products with N0 being the zeroth generation
    #     k_generations        - Storage for the k-values calculated across mtrials many number of trials
    #     debug                - A boolean flag that if true prints debug statements (WARNING: it's a LOT of print statements)
    #
    # The __init__ method also performs some error handling to ensure that class creation was completed correctly. 
    
    def __init__(self, N0, mtrials, shape_type, shape_parameters, purity, neutron_multiplicity, mean_free_path, max_generations, debug = False):   
        self.N0 = N0
        self.mtrials = mtrials
        self.shape_type = shape_type
        self.dimensions = {}
        self.volume = None
        self.surface_area = None
        self.shape_aspect_ratio = None
        self.purity = purity
        self.neutron_multiplicity = neutron_multiplicity
        self.mean_free_path = mean_free_path
        self.max_generations = max_generations
        self.k_generations = None
        self.debug = debug

        # Ensure that shape_type is an expected shape with implemented handling
        if self.shape_type not in Uranium235_Fission_Model.shapes:
            raise ValueError(f"Unsupported shape type {self.shape_type}")

        # Ensure that the purity of the sample is between 0 and 1
        if self.purity < 0 or self.purity > 1:
            raise ValueError(f"Expected Uranium purity between {0} and {1}:\nPurity given: {self.purity}")

        # Ensure that one and only one of "volume" and "dimensions" were used in shape construction
        if "volume" in shape_parameters and "dimensions" in shape_parameters:
            raise ValueError(f"Expected one of \"volume\" or \"dimensions\" input arguments.\nRecieved both.")

        # Ensures that if volume-based construction was used that the dimension parameters for each shape are handled accordingly.
        # "cube" and "sphere" models have volumes as a function of one variable; construction occurs as expected.
        # Volume-based construction for "cylinder" enforces the construction of a cylinder with radius r and height 2r.
        # Volume-based construction for "prism" enforces the construction of a square prism with length = l = width and height = 2l.
        if "volume" in shape_parameters:
            self.volume = shape_parameters["volume"]
            if shape_type == "sphere":
                self.shape_aspect_ratio = 1
                self.dimensions = {"radius": ((3 * self.volume)/(4 * np.pi))**(1/3)}
            elif shape_type == "cube":
                self.shape_aspect_ratio = 1
                self.dimensions = {"length": self.volume**(1/3)}
            elif shape_type == "cylinder":
                if "shape_aspect_ratio" not in shape_parameters or shape_parameters["shape_aspect_ratio"] <= 0:
                    raise ValueError(f"Argument shape_aspect_ratio > 0  expected for volume-based cylinder construction")
                self.shape_aspect_ratio = shape_parameters["shape_aspect_ratio"]
                radius = (self.volume/(shape_parameters["shape_aspect_ratio"] * np.pi))**(1/3)
                self.dimensions = {"radius": radius, "height": shape_parameters["shape_aspect_ratio"] * radius}
            elif shape_type == "prism":
                if "shape_aspect_ratio" not in shape_parameters or shape_parameters["shape_aspect_ratio"] <= 0:
                    raise ValueError(f"Argument shape_aspect_ratio > 0 expected for volume-based rectangular prism construction.")
                self.shape_aspect_ratio = shape_parameters["shape_aspect_ratio"]
                length = (self.volume * shape_parameters["shape_aspect_ratio"])**(1/3)
                self.dimensions = {"length": length, "width": length, "height": length/shape_parameters["shape_aspect_ratio"]}
            else:
                raise TypeError(f"Unexpected behaviour using volume-based model construction")

        # Ensure that if dimension-based construction was used that for the input shape_type that there is the expected number of 
        # dimension parameter-value pairs to define the shape. If the shape is properly defined store the dimensions in the class
        # variable and calculate the volume of the shape.
        elif "dimensions" in shape_parameters:
            expected_dimensions = Uranium235_Fission_Model.shapes[self.shape_type]["dimensions"]
            if len(shape_parameters["dimensions"]) != len(expected_dimensions):
                raise ValueError(f"Incorrect number of dimensions for {self.shape_type}.\n Expected {len(expected_dimensions)} dimensions: {expected_dimensions}")
            for param, value in shape_parameters["dimensions"].items():
                if param not in Uranium235_Fission_Model.shapes[self.shape_type]["dimensions"]:
                    raise ValueError(f"{param} is not a valid dimension for shape type {self.shape_type}")
                self.dimensions[param] = value
            if self.shape_type == "sphere":
                self.volume = (4/3) * np.pi * self.dimensions["radius"]**3
            elif self.shape_type == "cube":
                self.volume = self.dimensions["length"]**3
            elif self.shape_type == "cylinder":
                self.volume = np.pi * self.dimensions["radius"]**2 * self.dimensions["height"]
            elif self.shape_type == "prism":
                self.volume = self.dimensions["length"] * self.dimensions["width"] * self.dimensions["height"]
            else:
                raise TypeError(f"Unexpected behaviour using dimension-based model construction")

        # If all paremeters were input correctly and the __init__ function was able to populate all required values for
        # dimensions and volume then proceed to calculate the surface area of the shape.
        if shape_type == "cube":
            self.surface_area = 6 * (self.dimensions["length"]**2)
        elif shape_type == "sphere":
            self.surface_area = 4 * np.pi * (self.dimensions["radius"]**2)
        elif shape_type == "cylinder":
            self.surface_area = (2 * np.pi * self.dimensions["radius"]) * (self.dimensions["radius"] + self.dimensions["height"])
        elif shape_type == "prism":
            self.surface_area = 2 * (
                (self.dimensions["length"] * self.dimensions["width"]) + 
                (self.dimensions["length"] * self.dimensions["height"]) + 
                (self.dimensions["width"] * self.dimensions["height"])
            )
                
        # Call class function to generate randomized starting positions for zeroth generation of thermal neutrons based on class shape_type
        self.generate_N0()

    # Class Method - generate_N0
    # The class method generate_N0 is used to generate a randomly distributed (N0 x 3)-dimensional array of starting locations, r0, that are 
    # within the defined Uranium-235 sample shape. 
    def generate_N0(self):
        # Generate a (N0 x 3)-dimensional array of randomized starting positions within a Uranium-235 cube. Starting locations are calculated
        # for a cube in the 1st quadrant of the xyz-coordinate system.
        if self.shape_type == "cube":
            self.r0 = rdm.uniform(0, self.dimensions["length"], (self.N0, 3))

        # Generate a (N0 x 3)-dimensional array of randomized starting positions within a Uranium-235 sphere. Starting locations are calculated
        # using azimuth-altitude coordinates where phi is angle going CCW on xy-plane w.r.t +x-axis and theta is the angle above/below the
        # xy-plane.
        elif self.shape_type == "sphere":
            radius = self.dimensions["radius"]
            radii = radius * rdm.uniform(0, 1, self.N0)
            phi = rdm.uniform(0, 2 * np.pi, self.N0)
            cosphi = np.cos(phi)
            sinphi = np.sin(phi)
            costheta_random = rdm.uniform(-1, 1, self.N0)
            theta = np.arccos(costheta_random)
            costheta = np.cos(theta)
            sintheta = np.sin(theta)
            self.r0 = np.column_stack([radii*cosphi*sintheta, radii*sinphi*sintheta, radii*costheta])

        # Generate a (N0 x 3)-dimensional array of randomized starting positions within a Uranium-235 cylinder. Starting locations are calculated
        # for a cylinder where the boundary of the circle is in the xy-plane and the height of the cylinder runs along the z-axis.
        elif self.shape_type == "cylinder":
            heights = rdm.uniform(0, self.dimensions["height"], self.N0)
            radii = self.dimensions["radius"] * rdm.uniform(0, 1, self.N0)
            phi = rdm.uniform(0, 2 * np.pi, self.N0)
            self.r0 = np.column_stack([radii*np.cos(phi), radii*np.sin(phi), heights])

        # Generate a (N0 x 3)-dimensional array of randomized starting positions within a Uranium-235 prism. Starting locations are calculated
        # for a prism with length along the x-axis, width along the y-axis, and height along the z-axis.
        elif self.shape_type == "prism":
            lengths = rdm.uniform(0, self.dimensions["length"], self.N0)
            widths = rdm.uniform(0, self.dimensions["width"], self.N0)
            heights = rdm.uniform(0, self.dimensions["height"], self.N0)
            self.r0 = np.column_stack([lengths, widths, heights])

    # Class Method - captured
    # The class method captured takes in an (N x 3)-dimensional array r, representing the (x,y,z)-position of N-number of thermal neutrons, 
    # and determines: (1) how many thermal neutrons remained within the Uranium sample shape; (2) from those remaining within the sample, 
    # how many neutrons experience a fission event as a function of purity (where the probability of fission proportional to purity); and 
    # (3) how many new neutrons were produced by these fission events, dependent on the average number of new neutrons per fission event
    # that the model was constructed with.
    #     (1) This function first calls the appropriate shape's mask function to produce a boolean array representing the neutrons still
    #             within the Uranium sample. This then creates a masked array of N_1 neutrons, removing those that have escaped the sample.
    #     (2) A uniform randomly generated array of N_1 dimensions with values from [0,1) is created of fission probabilities, and using 
    #             the sample purity we then create a second boolean array of all values where sample purity > fission probability. Where
    #             the purity is greater than the probabilty a fission event occurs; where the purity is less than or equal to the probability
    #             no fission event will occur.
    #     (3) Finally a Poisson distributed random array, number_new_neutrons, with a mean defined by the model will be used to construct a 
    #             list new_neutrons where each neutron captured in r_capture r[i] is copied to the list number_new_neutrons[i] times.
    #             The list new_neutrons is recast as an array and returned by the method. This allows for models to be constructed with
    #             different number of average neutrons per fission even (such as U-235 2.4355).
    #             
    def captured(self, r):
        if self.debug:
            print(f"Next Position of Neutrons: \n{r}")

        # Call the appropriate shapes mask function to produce a boolean mask and remove neutrons that have escaped from array, leaving
        # only r_in the array of neutrons still in the sample.
        mask_function_name = Uranium235_Fission_Model.shapes[self.shape_type]["mask"]
        mask_function = getattr(self, mask_function_name)
        mask = mask_function(r)
        r_in = r[mask]

        # Create a uniformly distributed randomly generated array of values from 0 to 1. From this array a boolean array will be created
        # where true for all indices where sample purity p > probability generated. This mask will then be used to determine which of
        # remaining neutrons will undergo a fission event.
        fission_probability = rdm.uniform(0, 1, r_in.shape[0])
        r_captured = r_in[fission_probability < self.purity]

        # Create a Poisson distributed randomly generated N_2-dimensional array centered on the average neutrons created per fission 
        # event as set during the model construction (original testing at 2, U-235 average is 2.4355).
        number_new_neutrons = rdm.poisson(self.neutron_multiplicity, r_captured.shape[0])

        # Create new_neutrons array by using the Poisson distribution of numbers of neutrons created by each fission event
        # to determine how many neutrons each of the captured neutrons produces with the mean number created set during model construction
        new_neutrons = []
        for i, num in enumerate(number_new_neutrons):
            new_neutrons.extend([r_captured[i]] * num)
        new_neutrons = np.array(new_neutrons)
        
        if self.debug:
            print(f"Next Position of Absorbed Neutrons: \n{new_neutrons}")
            
        return new_neutrons

    # Class Method - next_gen
    # The class method next_gen is a recursive function that handles the bulk of each individual simulation. The method takes input 
    # arguments:
    #     r    - a (N x 3)-dimensional array representing the (x,y,z)-coordinate of N-many thermal neutrons within the sample shape
    #     gen  - an integer counter representing the current generation of fission products, beginning with the zeroth generation N0.
    # The recursion terminates if the generation counter meets or exceeds the model's set maximum generation or if the array of
    # neutrons becomes empty and the function returns an empty array and through each level of recursion the method then returns an
    # concatenated array of the prior levels return array and the current levels k multiplier.
    # The end return for the recursive function is an array of k multipliers from the first to last generation run by the simulation.
    def next_gen(self, r, gen):
        
        if gen >= self.max_generations or r.shape[0] == 0:
            return np.array([])
        
        # Generate random initial directions for each neutron 
        # Azimuth Angle Phi [0, 2pi) (xy-plane)
        phi = rdm.uniform(0, 2 * np.pi, r.shape[0])
                
        #Elevation Angle Theta from arccos([-1,1]) (angle below/above xy,plane)
        costheta_random = rdm.uniform(-1, 1, r.shape[0])
        theta = np.arccos(costheta_random)

        # Randomly generate the distance traveled for a neutron 
        # before it is absorbed. This comes from the probability 
        # p(L) ∝ e^{-L/λ} that a thermal neutron will be captured after
        # traveling a distance L
        d = rdm.exponential(mean_free_path, r.shape[0])

        # Calculate values for cos(phi) and sin(phi) to determine component distances
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        
        # Calculate values for cos(theta) and sin(theta) to determine component distances
        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        if self.debug:
            print(f"Current Neutron Number: {r.shape[0]}")
            print(f"Current Neutron Position: \n{r}")

        # Calculate the next position of each electron
        r_new = self.captured(r + np.column_stack([d*cosphi*sintheta, d*sinphi*sintheta, d*costheta]))

        if self.debug:
            k = r_new.shape[0]/r.shape[0]
            print(f"k-Factor: {k}")

        # Return a concatenates array containing this generations k multiplier value and a recursive
        # method call for the next iteration of positions and generation.
        return np.concatenate([np.array([r_new.shape[0]/r.shape[0]]), self.next_gen(r_new, gen+1)])

    # Class Method - dimensions_to_string
    # The class method dimensions_to_string iterates through the class dimensions dictionary and returns all
    # assigned key:value pairs for the dimensions in a formatted string.  This is used to provide an
    # informative and descriptive plot title for the histogram subplots.
    def dimensions_to_string(self):
        return "        ".join(f"{key.capitalize()} = {value:.3f} m" for key, value in self.dimensions.items())

    # Class method - truncate_sig_figs
    # The class function truncate_sig_figs is used to take the mean and standard deviation of a data set, determine how many significant
    # figures are present in the standard deviation and returns a formatted string in the form of mean plus/minus where each value is
    # rounded and truncated to the appropriate number of significant figures
    # NOTE: the Python documentation states that the inbuilt function round() rounds half values to the nearest round number. For instance
    #         the value 0.085 would round to 0.08, however 0.095 would round to 0.1. As such for values where the last significant figure
    #         is followed by the digit 5 with no other trailing figures, there may be some unexpected rounding. This has been left as is
    #         as we reasoned that the chances of the standard deviation being of the form 0. .... x500000.... would not be a frequent
    #         occurance
    def truncate_sig_figs(self, mean, standard_dev):
        if standard_dev == 0:
            return f"{mean:.0f} $\\pm$ {0}"
        decimal_number = int(np.abs(np.floor(np.log10(standard_dev))))
        magnitude = 10 ** -decimal_number
        std_trunc = int(round(standard_dev/magnitude)) * magnitude
        return f"{round(mean, decimal_number):.{decimal_number}f} $\\pm$ {std_trunc:.{decimal_number}f}" if (std_trunc/magnitude) < 10 else f"{round(mean, decimal_number - 1):.{decimal_number - 1}f} $\\pm$ {std_trunc:.{decimal_number - 1}f}"

    # Class Method - display_simulation_parameters
    # The class method display_simulation_parameters displays key simulation parameters such as initial neutron number N_0, number of trials, purity, neutron multiplicity, sample shape
    # and dimensions, SA:V ratio, SAR, and the mean multiplication factor k.
    # Displays a Latex formatted string using the IPython Display library with sublibraries display and Latex.
    # This information is included in the suptitle of the histogram plots and this method should only be called if plot_histograms is not called, otherwise redundant information will
    # be displayed.
    def display_simulation_parameters(self):
        display(Latex(f"Uranium-235 Fission Simulation and Sample Parameters:"))
        display(Latex(f"$N_0$ = {self.N0}{'    '}{'    '}$m_{{trials}}$ = {self.mtrials}{'    '}{'    '}Purity = {self.purity}{'    '}{'    '}Neutron Multiplicity = {self.neutron_multiplicity}{'    '}{'    '}$\\bar{{k}}$ = {self.truncate_sig_figs(self.k_generations.mean(), self.k_generations.std())}"))
        display(Latex(f"$^{{235}}U$ Sample Shape = {self.shape_type.capitalize()}{'    '}{'    '}{self.dimensions_to_string()}"))
        display(Latex(f"Volume = {self.volume:.3f} m$^3${'    '}{'    '}Shape-Aspect Ratio = {self.shape_aspect_ratio}{'    '}{'    '}Surface Area = {self.surface_area:.3f} m$^2${'    '}{'    '}Surface Area:Volume Ratio = {(self.surface_area/self.volume):.3f} m$^{{-1}}$"))
    
    # Class Method - plot_histograms
    # The class method plot_histogram expects no input arguments and uses data from the class variable k_generations
    # that stores each k-value across the iterated generations.
    # The method creates a histogram subplot showing the distribution of k multipliers per generation group.
    # The group of subplots has a title to display model settings and testing parameters and each generation
    # subplot contains data about the mean and standard deviation across that generation
    def plot_histograms(self):

        # Instantiate figure and ax, flatten ax for easier loop iteration when adding histograms to the subplot
        num_rows = int(np.ceil(self.k_generations.shape[1]/2))
        fig, ax = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
        ax = ax.flatten()

        # Sets a descriptive title string for the title of the group of subplots
        suptitle_string = f"""Uranium-235 Fission Simulation and Sample Parameters:\n
                                $N_0$ = {self.N0}{'    '}{'    '}$m_{{trials}}$ = {self.mtrials}{'    '}{'    '}Purity = {self.purity}{'    '}{'    '}Neutron Multiplicity = {self.neutron_multiplicity}{'    '}{'    '}$\\bar{{k}}$ = {self.truncate_sig_figs(self.k_generations.mean(), self.k_generations.std())}\n
                                $^{{235}}U$ Sample Shape = {self.shape_type.capitalize()}{'    '}{'    '}{self.dimensions_to_string()}\n
                                Volume = {self.volume:.3f} m$^3${'    '}{'    '}Shape-Aspect Ratio = {self.shape_aspect_ratio}{'    '}{'    '}Surface Area = {self.surface_area:.3f} m$^2${'    '}{'    '}Surface Area:Volume Ratio = {(self.surface_area/self.volume):.3f} m$^{{-1}}$\n
        """
        fig.suptitle(suptitle_string, x=0.5)

        # Create a subplot with a histogram for frequency of k multiplier across testing for a single generation group. 
        for gen in range(self.k_generations.shape[1]):
            ax[gen].hist(self.k_generations[:, gen], bins=20, alpha=0.7, color='blue', edgecolor='black')
            ax[gen].set_title(f"k(Gen. {gen} to {gen+1}) = {self.k_generations[:,gen].mean():.3f} ± {self.k_generations[:,gen].std():.3f}")
            ax[gen].set_xlabel('Multiplication Factor, k')
            ax[gen].set_ylabel('Frequency')
            ax[gen].grid(True)

        # Remove an extra subplots that were created but not used
        if self.k_generations.shape[1] < len(ax):
            for i in range(self.k_generations.shape[1], len(ax)):
                fig.delaxes(ax[i])

        # Layout and show
        plt.tight_layout()
        plt.show()

    # Class Method - plot_k_versus_volume
    # The class method plot_k_versus_volume is a static method called from the class itself rather than an
    # instantiation of the class; as such, it does not have access to specific instance variables.
    # plot_k_versus_volume expects input arguments of
    #     volume             - an array of volumes for each volume of shape tested within a volume phase-space analysis
    #     SAtoV              - an array of surface area to volume ratios for each different volume of shape tested
    #     k_generations_mean - a mean k-value for each generation of simulation for each different volume of shape tested
    # The class then creates a subplot where the first graph is a surface graph showing how the mean k-value varied across
    # each generation and each shape volume tested, where mean k-value is a function of generation and volume. The second
    # subplot is a 2D plot of surface area to volume by volume for each volume tested.
    def plot_k_versus_volume(sample_shape, volume, SAtoV, k_generations_mean):
 
        generations = np.linspace(1, k_generations_mean.shape[1], k_generations_mean.shape[1], endpoint=True)
        VOLUME_grid, GEN_grid = np.meshgrid(volume, generations)

        suptitle_string = f"""Uranium-235 Fission Simulation:\n
                                Volume Phase-Space Analysis of {sample_shape.capitalize()}-Shaped $^{{235}}U$ Fuel Sample\n\n                      
        """
        
        fig = plt.figure(figsize=(16, 8))
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

        fig.suptitle(suptitle_string, x=0.5)
        
        # Surface plot
        heatmap = ax1.pcolormesh(VOLUME_grid, GEN_grid, k_generations_mean.T, cmap='viridis', edgecolor='k', shading='auto')

        # Adjust ticks on the volumes axis to be set to the exact values within volume
        ax1.set_xticks(volume)
        
        # Adjust ticks on the generation axis
        ax1.set_yticks(np.arange(int(min(generations)), int(max(generations))+1, 1))

        
        # Add labels and colorbar
        ax1.set_title(f'Mean $k$ Factor versus Volume and Generation Number')
        ax1.set_xlabel(f'Volume, V [m$^3$]')
        ax1.set_ylabel(f'Generation Number, N')
        ax1.autoscale(tight=True)
        fig.colorbar(heatmap, shrink=0.5, aspect=10, pad=0.15, label='Mean k Value')

        # 2D plot of SA/V versus volume
        ax2.plot(volume, SAtoV, 'b--', label = "SA:V versus V")
        
        ax2.set_title(f'Surface Area to Volume Ratio versus Volume')
        ax2.set_xlabel(f'Volume, V [m$^3$]')
        ax2.set_ylabel(f'Surface Area to Volume Ratio, SA:V [m$^{-1}$]')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    # Class Method - plot_k_versus_shape_aspect_ratio
    # The class method plot_k_versus_shape_aspect_ratio is a static method called from the class itself rather than an
    # instantiation of the class; as such, it does not have access to specific instance variables.
    # plot_k_versus_volume expects input arguments of
    #     SAR                - an array of shape aspect ratios (SAR) for each SAR of shape tested within a SAR phase-space analysis
    #     SAtoV              - an array of surface area to volume ratios for each different volume of shape tested
    #     k_generations_mean - a mean k-value for each generation of simulation for each different volume of shape tested
    # The class then creates a subplot where the first graph is a surface graph showing how the mean k-value varied across
    # each generation and each shape SAR tested, where mean k-value is a function of generation and SAR. The second
    # subplot is a 2D plot of surface area to volume (SAtoV) by SAR for each SAR tested to show how the SAtoV of the shape changed
    # with respect to SAR.
    def plot_k_versus_shape_aspect_ratio(sample_shape, SAR, SAtoV, k_generations_mean):
 
        generations = np.linspace(1, k_generations_mean.shape[1], k_generations_mean.shape[1], endpoint=True)
        SAR_grid, GEN_grid = np.meshgrid(SAR, generations)

        suptitle_string = f"""Uranium-235 Fission Simulation:\n
                                Shape Aspect Ratio Phase-Space Analysis of {sample_shape.capitalize()}-Shaped $^{{235}}U$ Fuel Sample\n\n                        
        """
        
        fig = plt.figure(figsize=(16, 8))
        ax1, ax2 = fig.add_subplot(121), fig.add_subplot(122)

        fig.suptitle(suptitle_string, x=0.5)
        
        # Surface plot
        heatmap = ax1.pcolormesh(SAR_grid, GEN_grid, k_generations_mean.T, cmap='viridis', edgecolor='k', shading='auto')

        # Adjust ticks on the SAR axis
        ax1.set_xticks(np.arange(min(SAR), max(SAR)+1, 1))
    
        # Adjust ticks on the generation axis
        ax1.set_yticks(np.arange(int(min(generations)), int(max(generations))+1, 1))
            
        # Add labels and colorbar
        ax1.set_title(f"Mean $k$ Factor versus Shape Aspect Ratio and Generation Number")
        ax1.set_xlabel('Shape Aspect Ratio, SAR')
        ax1.set_ylabel('Generation Number, N')
        fig.colorbar(heatmap, ax=ax1, shrink=0.5, aspect=10, pad=0.15, label='Mean k Value')

        ax2.plot(SAR, SAtoV, 'b--', label = "SA:V versus SAR")
        
        ax2.set_title(f'Surface Area to Volume Ratio versus Shape Aspect Ratio')
        ax2.set_xlabel(f'Shape Aspect Ratio, SAR ')
        ax2.set_ylabel(f'Surface Area to Volume Ratio, SA:V [m$^{-1}$]')
        ax2.grid(True)
        ax2.legend()

    # Class Method - captured
    # The class method captured takes in an (N x 3)-dimensional array r, representing the (x,y,z)-position of N-number of thermal neutrons, 
    # and determines: (1) how many thermal neutrons remained within the Uranium sample shape; (2) from those remaining within the sample, 
    # it multiplies each neutron by the neutron multiplicity value to generate new neutrons for the fission event. THIS METHOD IS ONLY TO
    # BE USED TO VALIDATE TRAJECTORY BEHAVIOUR - RANDOMIZED DISTRIBUTION FOR NEUTRON MULTIPLICITY AND PURITY VALUE DISABLED IN THIS METHOD
    # AS THIS METHOD IS INTENDED TO BE USED ON A SMALL NUMBER OF NEUTRONS ONLY!
    #     (1) This function first calls the appropriate shape's mask function to produce a boolean array representing the neutrons still
    #             within the Uranium sample. This then creates a masked array of N_1 neutrons, removing those that have escaped the sample.
    #     (2) An array of length equal to the number of neutrons in this generation is created filled with values equal to the neutron multiplicity
    #             and the neutrons within the sample are then replicated that many times for each neutron. The randomized Poisson distribution 
    #             used in this method has been disabled as we are working with a small sample of neutrons for which a distribution doesn't make
    #             sense. Instead, each neutron is guaranteed to be multiplied by the neutron multiplicity, where the neutron multiplicity is expected
    #             to be an integer value.
    # The only changes between method captured and captured_trajectories is the removal of sample purity affecting fission probability and the
    # Poisson distribution for amount of neutrons produced for each event, as this is intended for a small sample for which distributions are not
    # appropriate and do not fit the purpose, as this is intended to test trajectories. However the functionality outside of this has not been
    # altered otherwise, and as such this will still be able to verify the behaviour of the model with respect to neutron trajectories.
    def captured_trajectories(self, r):
        
        if self.debug:
            print(f"Next Position of Neutrons: \n{r}")

        # Call the appropriate shapes mask function to produce a boolean mask and remove neutrons that have escaped from array, leaving
        # only r_in the array of neutrons still in the sample.
        mask_function_name = Uranium235_Fission_Model.shapes[self.shape_type]["mask"]
        mask_function = getattr(self, mask_function_name)
        mask = mask_function(r)
        r_in = r[mask]

        # Create an array of length len(r_in) populated with the integer value of neutron multiplicity in order to multiply each
        # respective neutron for the fission event.
        number_new_neutrons = np.full(r_in.shape[0], self.neutron_multiplicity, dtype=int)

        # Create new_neutrons array by using the number_new_neutrons array
        new_neutrons = []
        for i, num in enumerate(number_new_neutrons):
            new_neutrons.extend([r_in[i]] * num)
        new_neutrons = np.array(new_neutrons)
        
        if self.debug:
            print(f"Next Position of Absorbed Neutrons: \n{new_neutrons}")
            
        return new_neutrons

    # Class Method - next_gen_trajectories
    # The class method next_gen_trajectories is a recursive function that handles the bulk of each individual simulation FOR THE PURPOSE
    # OF PLOTTING TRAJECTORY ONLY! This is not the correct method to run the simulation, this is only for verifying that the neutron
    # trajectories are behaving as desired. The method takes input arguments:
    #     r    - a (N x 3)-dimensional array representing the (x,y,z)-coordinate of N-many thermal neutrons within the sample shape
    #     gen  - an integer counter representing the current generation of fission products, beginning with the zeroth generation N0.
    # The recursion terminates if the generation counter meets or exceeds the model's set maximum generation or if the array of
    # neutrons becomes empty and the function returns the position of each neutron as well as it's direction vector for each generation
    # and the number of neutrons in each generation as a list. These lists are then used to plot the trajectories across the generations 
    # for behaviour validation.
    def next_gen_trajectories(self, r, gen):
    
        if gen >= self.max_generations or r.shape[0] == 0:
            d = np.zeros(r.shape[0])
            d_vec = np.column_stack([d, d, d])
            return [],[],[]
        
        # Generate random initial directions for each neutron 
        # Azimuth Angle Phi [0, 2pi) (xy-plane)
        phi = rdm.uniform(0, 2 * np.pi, r.shape[0])
                
        #Elevation Angle Theta [-pi, pi] (angle below/above xy,plane)
        costheta_random = rdm.uniform(-1, 1, r.shape[0])
        theta = np.arccos(costheta_random)

        # Randomly generate the distance traveled for a neutron 
        # before it is absorbed. This comes from the probability 
        # p(L) ∝ e^{-L/λ} that a thermal neutron will be captured after
        # traveling a distance L
        d = rdm.exponential(mean_free_path, r.shape[0])

        # Calculate values for cos(phi) and sin(phi) to determine component distances
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        
        # Calculate values for cos(theta) and sin(theta) to determine component distances
        sintheta = np.sin(theta)
        costheta = np.cos(theta)

        if self.debug:
            print(f"Current Neutron Number: {r.shape[0]}")
            print(f"Current Neutron Position: \n{r}")

        # Calculate distance vector
        d_vec = np.column_stack([d*cosphi*sintheta, d*sinphi*sintheta, d*costheta])

        r_new_preprocessed = r + d_vec

        # Create a uniformly distributed randomly generated array of values from 0 to 1. From this array a boolean array will be created
        # where true for all indices where sample purity p > probability generated. This mask will then be used to determine which of
        # remaining neutrons will undergo a fission event.
        fission_probability = rdm.uniform(0, 1, r_new_preprocessed.shape[0])
        
        r_new_preprocessed_prob = r_new_preprocessed[fission_probability < self.purity]
        
        r_captured = r_new_preprocessed[fission_probability < self.purity]

        # Calculate the next position of each electron
        r_new = self.captured_trajectories(r_captured)


        # Get the next position vector, direction vector, and an list of integers representing which generation it is
        # and the length of the list representing how many neutrons are in each generation, to be used for properly
        # color formatting the trajectory plots
        next_position, next_direction, next_gen_count = self.next_gen_trajectories(r_new, gen+1)

        if self.debug:
            k = r_new.shape[0]/r.shape[0]
            print(f"k-Factor: {k}")

        # Set the position, direction and generation count lists
        positions = [r] + next_position
        directions = [d_vec] + next_direction
        gen_count = [np.full(r.shape[0], gen)] + next_gen_count

        return positions, directions, gen_count


    # Class Method - plot_trajectories
    # The class method plot_trajectories is used to run a single simulation, intended for a small number of initial neutrons in order to 
    # validate the behaviour of the model with respect to the neutron trajectories.
    # Calling this method will run the model as it was set up, however rather than collecting/storing k values it will collect an array of
    # positions and direction vectors in order to plot the trajectory of each neutron across several generations of fission events.
    # plot_trajectories displays:
    #     3D quiver plot of the trajectories in 3D space.
    #     2D quiver projections onto each of the XY-plane, XZ-plane and YZ-plane in order to better visualize the movement and help
    #         with any losses in clarity due to perspective with the 3D plot.
    # The method ensures that the argument for neutron multiplicity is of integer value, and if correctly initialized will run the simulation
    # and plot the trajectories.
    # NOTE: THIS IS ONLY TO BE USED FOR TRAJECTORY VALIDATION AND DOES NOT INCLUDE MUCH OF THE FUNCTIONALITY FOR INVESTIGATION INTO K-VALUES
    # OR PHASE SPACES.
    def plot_trajectories(self):

        # Verifies the model correctly instantiated with integer value for neutron multiplicity.
        if not isinstance(self.neutron_multiplicity, int):
            raise TypeError(f"For testing trajectory neutron multiplicity expected to be an integer value:\nRecieved {self.neutron_multiplicity} of dtype {type(self.neutron_multiplicity)}")

        # Calls recursive function to run model across all generations.
        positions, directions, gen_count = self.next_gen_trajectories(self.r0, 0)
    
        # Sets positions, directions and gen_count lists ensuring no empty lists included
        positions = [p for p in positions if p.shape[0] > 0]
        directions = [d for d in directions if d.shape[0] > 0]
        gen_count = [g for g in gen_count if g.shape[0] > 0]

        # 
        positions = np.vstack(positions)
        directions = np.vstack(directions)
        gen_count = np.hstack(gen_count)
    
        # Create a discrete colormap with rainbow colours for plotting trajectories across multiple generations
        base_cmap = colormaps["rainbow"]
        discrete_cmap = ListedColormap(base_cmap(np.linspace(0, 1, int(np.max(gen_count) + 1))))

        # Create figure
        fig = plt.figure(figsize=(15, 15))
    
        # Create subplots
        ax3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_xy = fig.add_subplot(2, 2, 2)
        ax_xz = fig.add_subplot(2, 2, 3)
        ax_yz = fig.add_subplot(2, 2, 4)
    
    
        # Iterate over each generation
        for gen in np.unique(gen_count):
            
            gen_indices = np.where(gen_count == gen)[0]
            color = discrete_cmap(gen)
    
            # 3D quiver plot for neutron trajectores in R^3
            ax3d.quiver(
                positions[gen_indices, 0], positions[gen_indices, 1], positions[gen_indices, 2],
                directions[gen_indices, 0], directions[gen_indices, 1], directions[gen_indices, 2],
                color=color, normalize=False, arrow_length_ratio=0.1
            )

            # Scatter points for neutron positions in 3D plot
            ax3d.scatter(
                positions[gen_indices, 0], positions[gen_indices, 1], positions[gen_indices, 2],
                color=color, s=30, marker='o', label=f"Gen {gen}" if gen == np.unique(gen_count)[0] else ""
            )
    
            # XY-plane projection
            ax_xy.quiver(
                positions[gen_indices, 0], positions[gen_indices, 1],
                directions[gen_indices, 0], directions[gen_indices, 1],
                color=color, angles="xy", scale_units="xy", scale=1
            )

            # Scatter points for neutron positions in XY-plane
            ax_xy.scatter(
                positions[gen_indices, 0], positions[gen_indices, 1],
                color=color, s=30, marker='o', label=f"Gen {gen}" if gen == np.unique(gen_count)[0] else ""
            )

    
            # XZ-plane projection
            ax_xz.quiver(
                positions[gen_indices, 0], positions[gen_indices, 2],
                directions[gen_indices, 0], directions[gen_indices, 2],
                color=color, angles="xy", scale_units="xy", scale=1
            )

            # Scatter points for neutron positions in XZ-plane
            ax_xz.scatter(
                positions[gen_indices, 0], positions[gen_indices, 2],
                color=color, s=30, marker='o', label=f"Gen {gen}" if gen == np.unique(gen_count)[0] else ""
            )

    
            # YZ-plane projection
            ax_yz.quiver(
                positions[gen_indices, 1], positions[gen_indices, 2],
                directions[gen_indices, 1], directions[gen_indices, 2],
                color=color, angles="xy", scale_units="xy", scale=1
            )

            # Scatter points for neutron positions in YZ plane
            ax_yz.scatter(
                positions[gen_indices, 1], positions[gen_indices, 2],
                color=color, s=30, marker='o', label=f"Gen {gen}" if gen == np.unique(gen_count)[0] else ""
            )

        # Create colorbar specifically for ax3d
        sm = plt.cm.ScalarMappable(cmap=discrete_cmap, norm=plt.Normalize(vmin=np.min(gen_count), vmax=np.max(gen_count)))
        sm.set_array([])
        
        # Create a linspace of integers from 0 to the maximum generation count
        gen_ticks = np.linspace(0, np.max(gen_count), num=int(np.max(gen_count)) + 1, dtype=int)

        # Set the colorbar parameters and ticks
        cbar = fig.colorbar(sm, ax=ax3d, orientation='vertical', fraction=0.02, pad=0.1)
        cbar.set_ticks(gen_ticks)
        cbar.set_label("Generation Count")
        
        # Define boundaries based on min/max positions (consider a small margin)
        margin = 0.05 * self.dimensions["length"]

        # Wireframe cube for the 3D plot
        wireframe_x = [0, self.dimensions["length"]]
        wireframe_y = [0, self.dimensions["length"]]
        wireframe_z = [0, self.dimensions["length"]]
    
        # Wireframe for 3D space
        for x in wireframe_x:
            for y in wireframe_y:
                for z in wireframe_z:
                    ax3d.plot([x, x], [y, y], [0, self.dimensions["length"]], color='k')
                    ax3d.plot([x, x], [0, self.dimensions["length"]], [z, z], color='k')
                    ax3d.plot([0, self.dimensions["length"]], [x, x], [z, z], color='k')

        # Draw the boundary for each 2D projection (0 to L for each axis)
        # XY plane boundary
        ax_xy.plot([0, self.dimensions["length"]], [0, 0], color='black')  # Bottom
        ax_xy.plot([0, self.dimensions["length"]], [self.dimensions["length"], self.dimensions["length"]], color='black')  # Top
        ax_xy.plot([0, 0], [0, self.dimensions["length"]], color='black')  # Left
        ax_xy.plot([self.dimensions["length"], self.dimensions["length"]], [0, self.dimensions["length"]], color='black')  # Right
    
        # XZ plane boundary
        ax_xz.plot([0, self.dimensions["length"]], [0, 0], color='black')  # Bottom
        ax_xz.plot([0, self.dimensions["length"]], [self.dimensions["length"], self.dimensions["length"]], color='black')  # Top
        ax_xz.plot([0, 0], [0, self.dimensions["length"]], color='black')  # Left
        ax_xz.plot([self.dimensions["length"], self.dimensions["length"]], [0, self.dimensions["length"]], color='black')  # Right
    
        # YZ plane boundary
        ax_yz.plot([0, self.dimensions["length"]], [0, 0], color='black')  # Bottom
        ax_yz.plot([0, self.dimensions["length"]], [self.dimensions["length"], self.dimensions["length"]], color='black')  # Top
        ax_yz.plot([0, 0], [0, self.dimensions["length"]], color='black')  # Left
        ax_yz.plot([self.dimensions["length"], self.dimensions["length"]], [0, self.dimensions["length"]], color='black')  # Right

        
        # Set plot titles and labels        
        ax3d.set_title("3D Trajectories")
        ax3d.set_xlabel("X-axis [m]")
        ax3d.set_ylabel("Y -axis [m]")
        ax3d.set_zlabel("Z-axis [m]")
        ax3d.set_xlim([0 - margin, self.dimensions["length"] + margin])
        ax3d.set_ylim([0 - margin, self.dimensions["length"] + margin])
        ax3d.set_zlim([0 - margin, self.dimensions["length"] + margin])
    
        ax_xy.set_title("XY-Plane Projection")
        ax_xy.set_xlabel("X-axis [m]")
        ax_xy.set_ylabel("Y-axis [m]")
        ax_xy.set_xlim([0 - margin, self.dimensions["length"] + margin])
        ax_xy.set_ylim([0 - margin, self.dimensions["length"] + margin])
    
        ax_xz.set_title("XZ-Plane Projection")
        ax_xz.set_xlabel("X-axis [m]")
        ax_xz.set_ylabel("Z-axis [m]")
        ax_xz.set_xlim([0 - margin, self.dimensions["length"] + margin])
        ax_xz.set_ylim([0 - margin, self.dimensions["length"] + margin])
    
        ax_yz.set_title("YZ-Plane Projection")
        ax_yz.set_xlabel("Y-axis [m]")
        ax_yz.set_ylabel("Z-axis [m]")
        ax_yz.set_xlim([0 - margin, self.dimensions["length"] + margin])
        ax_yz.set_ylim([0 - margin, self.dimensions["length"] + margin])
    
        # Adjust layout and add colorbar
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.25, wspace=0.25)  # Adjust spacing between subplots

        plt.show()
        

    # Class Method - run_simulation
    # The class method run_simulation runs a single simulation using the initial parameters the system was constructed with.
    # This class calls the first layer of the next_gen recursive method and accepts the returned array of k multipliers
    # from next_gen. This function also returns that Dane same array to be used for plotting data in other methods.
    def run_simulation(self):
        k_arr = self.next_gen(self.r0, 0)
        if self.debug:
            print(k_arr)
        return k_arr

    # Class Method - run_trials
    # The class method run_trials calls the run_simulation method m-many times for mtrials, appending each return array into
    # a temporary list. When the trials are completed the k_list is copied to an array and sent to the plot_histogram method.
    def run_trials(self):
        k_list = []

        # Run the simulation m-many times and append the returned array of k-values to the k_list list. When complete convert
        # k_list into an array k_trials
        for i in range(self.mtrials):
            k_list.append(self.run_simulation())

        # Pad each array of k-values within k_list with 0 up to the set length of max_generations, if not already sufficient length
        k_list_padded = [np.pad(k_arr, (0, self.max_generations - len(k_arr))) for k_arr in k_list]
        
        k_trials = np.array(k_list_padded)  # Now, this should work without the shape error

        self.k_generations = k_trials


###############################################################################################################################################

print(f"Initialized.")