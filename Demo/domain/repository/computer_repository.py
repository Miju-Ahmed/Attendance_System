from abc import ABC, abstractmethod

from ..model import Computer

class ComputerRepository(ABC):
    @abstractmethod
    def add_computer(self, computer: Computer) -> Computer:
        raise NotImplementedError("Implement add_computer method")
    
    @abstractmethod
    def update_computer(self, computer: Computer) -> Computer:
        raise NotImplementedError("Implement update_computer method")
    
    @abstractmethod
    def get_computer_by_id(self, computer_id: str) -> Computer:
        raise NotImplementedError("Implement get_computer_by_id method")

    @abstractmethod
    def delete_computer(self, computer_id: str) -> bool:
        raise NotImplementedError("Implement delete_computer method")
