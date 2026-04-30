from dataclasses import dataclass, asdict, field

@dataclass
class Tissue:
    """
    A dataclass representing a tissue in the body.

    Attributes:
    -----------
    name : str, optional
        The name of the tissue (e.g. blood, myocardium, bone, etc.)
    c : float
        The speed of sound in the tissue (in m/s)
    rho : float
        The density of the tissue (in kg/m^3)
    sigma : float
        The standard deviation of speed of sound in the tissue (in m/s)
    scale : float
        The scale factor for tissue heterogeneity
    label : int, optional
        A label for the tissue (an integer between 0 and n)
    """
    name: str = None
    c: float = 1540
    rho: float = 1000
    sigma: float = 1
    scale: float = 0.1
    label: int = None

    def save(self):
        """
        Save the tissue attributes to a dictionary.
        
        Returns:
        --------
        dict
            A dictionary containing the tissue attributes.
        """
        return asdict(self)

    def load(self, dictionary):
        """
        Load tissue attributes from a dictionary.
        
        Parameters:
        -----------
        dictionary : dict
            A dictionary containing tissue attributes.
            
        Returns:
        --------
        Tissue
            The tissue object with updated attributes.
        """
        for key, value in dictionary.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
