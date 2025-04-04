import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
import re
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

@dataclass
class StaffState:
    """Tracks the current state of a staff including clef and key signature"""
    staff_id: int
    clef_type: str = "gClef"  # Default to treble clef
    key_signature: str = ""   # E.g., "BMajor", "FMinor", etc.
    
    # Keep track of active accidentals (reset at each measure)
    active_accidentals: Dict[str, str] = None
    
    def __post_init__(self):
        if self.active_accidentals is None:
            self.active_accidentals = {}

@dataclass
class Note:
    """Represents a note with its position and pitch information"""
    id: str
    class_name: str
    staff_id: int
    line_pos: str
    midi_pitch: Optional[int] = None
    pitch_step: Optional[str] = None
    pitch_octave: Optional[int] = None
    top: int = 0
    left: int = 0
    width: int = 0
    height: int = 0
    onset_time: float = 0.0
    # Add fields for interpretation
    interpreted_pitch: Optional[str] = None


@dataclass
class Accidental:
    """Represents an accidental and its position"""
    id: str
    class_name: str  # "accidentalSharp", "accidentalFlat", "accidentalNatural", etc.
    staff_id: int
    line_pos: str
    top: int = 0
    left: int = 0
    width: int = 0
    height: int = 0


@dataclass
class KeySignature:
    """Represents a key signature"""
    id: str
    class_name: str  # "BMajor", "FMinor", etc.
    staff_id: int
    description: Optional[str] = None
    
    # Calculate affected pitches based on key signature
    @property
    def affected_pitches(self) -> Dict[str, str]:
        """Returns a dictionary mapping pitch letters to accidentals"""
        # Circle of fifths: order of sharps (F C G D A E B) and flats (B E A D G C F)
        sharps_order = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
        flats_order = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
        
        affected = {}
        
        # Extract major/minor and key from class name or description
        key_info = self.description if self.description else self.class_name
        
        # Handle different formats: "BMajor", "F#Minor", "B Major", etc.
        match = re.match(r'([A-G][b#]?)(?:\s*)(Major|Minor|Maj|Min)', key_info, re.IGNORECASE)
        if not match:
            return affected
            
        root, mode = match.groups()
        mode = mode.lower()
        
        # Determine number of sharps/flats based on key
        # Simplified calculation (would need to be expanded for all keys)
        if 'major' in mode or 'maj' in mode:
            if root == 'C':
                return {}  # C Major has no accidentals
            elif root == 'G':
                affected = {sharps_order[0]: '#'}  # 1 sharp (F#)
            elif root == 'D':
                affected = {p: '#' for p in sharps_order[:2]}  # 2 sharps (F# C#)
            elif root == 'A':
                affected = {p: '#' for p in sharps_order[:3]}  # 3 sharps
            elif root == 'E':
                affected = {p: '#' for p in sharps_order[:4]}  # 4 sharps
            elif root == 'B':
                affected = {p: '#' for p in sharps_order[:5]}  # 5 sharps
            elif root == 'F#':
                affected = {p: '#' for p in sharps_order[:6]}  # 6 sharps
            elif root == 'C#':
                affected = {p: '#' for p in sharps_order}      # 7 sharps
            elif root == 'F':
                affected = {flats_order[0]: 'b'}  # 1 flat (Bb)
            elif root == 'Bb':
                affected = {p: 'b' for p in flats_order[:2]}  # 2 flats
            elif root == 'Eb':
                affected = {p: 'b' for p in flats_order[:3]}  # 3 flats
            elif root == 'Ab':
                affected = {p: 'b' for p in flats_order[:4]}  # 4 flats
            elif root == 'Db':
                affected = {p: 'b' for p in flats_order[:5]}  # 5 flats
            elif root == 'Gb':
                affected = {p: 'b' for p in flats_order[:6]}  # 6 flats
            elif root == 'Cb':
                affected = {p: 'b' for p in flats_order}      # 7 flats
        
        # Similar logic for minor keys (offset in circle of fifths)
        # Would need to complete for all minor keys
        
        return affected


class EnhancedPitchMapper:
    """
    Advanced analyzer that maps staff positions to pitches accounting for
    clefs, key signatures, accidentals, and note ordering.
    """
    
    def __init__(self):
        # Base pitch mapping for different clefs (position -> pitch letter)
        self.base_pitch_maps = {
            "gClef": {  # Treble clef
                "L1": "E", "L2": "G", "L3": "B", "L4": "D", "L5": "F",
                "S1": "F", "S2": "A", "S3": "C", "S4": "E", "S0": "D"
            },
            "fClef": {  # Bass clef
                "L1": "G", "L2": "B", "L3": "D", "L4": "F", "L5": "A",
                "S1": "A", "S2": "C", "S3": "E", "S4": "G", "S0": "F"
            },
            "cClef": {  # Alto clef (C on 3rd line)
                "L1": "F", "L2": "A", "L3": "C", "L4": "E", "L5": "G",
                "S1": "G", "S2": "B", "S3": "D", "S4": "F", "S0": "E"
            },
            # Add more clef types as needed
        }
        
        # Default octaves for each clef
        self.base_octave_maps = {
            "gClef": {
                "L1": 4, "L2": 4, "L3": 4, "L4": 5, "L5": 5,
                "S1": 4, "S2": 4, "S3": 5, "S4": 5, "S0": 4
            },
            "fClef": {
                "L1": 2, "L2": 2, "L3": 3, "L4": 3, "L5": 3,
                "S1": 2, "S2": 3, "S3": 3, "S4": 3, "S0": 2
            },
            "cClef": {
                "L1": 3, "L2": 3, "L3": 4, "L4": 4, "L5": 4,
                "S1": 3, "S2": 3, "S3": 4, "S4": 4, "S0": 3
            }
        }
        
        # Extension mappings for positions beyond the standard staff
        self.extensions = {
            "above": ["A", "B", "C", "D", "E", "F", "G"],  # Pitches cycling upward
            "below": ["C", "B", "A", "G", "F", "E", "D"]   # Pitches cycling downward
        }
        
        # Track staff states
        self.staff_states = {}
        
        # Store all music elements by staff
        self.staff_elements = defaultdict(list)
        
        # Store barlines to track measures
        self.barlines = []
        
        # Results of analysis
        self.interpreted_notes = []
    
    def process_xml(self, xml_file):
        """Process an XML file to identify staff elements and their positions"""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Process all nodes
            nodes = root.findall('.//Node')
            
            for node in nodes:
                class_name = node.find('ClassName')
                class_name = class_name.text if class_name is not None else ""
                
                # Skip nodes without class name
                if not class_name:
                    continue
                
                # Extract basic information
                node_id = node.find('Id').text if node.find('Id') is not None else ""
                top = int(node.find('Top').text) if node.find('Top') is not None else 0
                left = int(node.find('Left').text) if node.find('Left') is not None else 0
                width = int(node.find('Width').text) if node.find('Width') is not None else 0
                height = int(node.find('Height').text) if node.find('Height') is not None else 0
                
                # Get data items
                data_items = {}
                for item in node.findall('.//DataItem'):
                    key = item.get('key')
                    value = item.text
                    value_type = item.get('type')
                    
                    # Convert value based on type
                    if value_type == 'int':
                        value = int(value)
                    elif value_type == 'float':
                        value = float(value)
                        
                    data_items[key] = value
                
                # Skip if we don't have staff_id
                if 'staff_id' not in data_items:
                    continue
                
                staff_id = data_items['staff_id']
                
                # Process node based on class
                if 'Clef' in class_name:  # gClef, fClef, cClef etc.
                    # Update staff state with clef
                    if staff_id not in self.staff_states:
                        self.staff_states[staff_id] = StaffState(staff_id=staff_id)
                    
                    self.staff_states[staff_id].clef_type = class_name
                
                elif 'Major' in class_name or 'Minor' in class_name:
                    # Key signature
                    key_signature = KeySignature(
                        id=node_id,
                        class_name=class_name,
                        staff_id=staff_id,
                        description=data_items.get('key_signature_description')
                    )
                    
                    # Update staff state with key signature
                    if staff_id not in self.staff_states:
                        self.staff_states[staff_id] = StaffState(staff_id=staff_id)
                    
                    self.staff_states[staff_id].key_signature = class_name
                    
                    # Store element
                    self.staff_elements[staff_id].append(key_signature)
                
                elif 'accidental' in class_name.lower():
                    # Accidental
                    accidental = Accidental(
                        id=node_id,
                        class_name=class_name,
                        staff_id=staff_id,
                        line_pos=data_items.get('line_pos', ''),
                        top=top,
                        left=left,
                        width=width,
                        height=height
                    )
                    
                    # Store element
                    self.staff_elements[staff_id].append(accidental)
                
                elif 'notehead' in class_name.lower():
                    # Note
                    note = Note(
                        id=node_id,
                        class_name=class_name,
                        staff_id=staff_id,
                        line_pos=data_items.get('line_pos', ''),
                        midi_pitch=data_items.get('midi_pitch_code'),
                        pitch_step=data_items.get('normalized_pitch_step'),
                        pitch_octave=data_items.get('pitch_octave'),
                        top=top,
                        left=left,
                        width=width,
                        height=height,
                        onset_time=data_items.get('onset_beats', 0.0)
                    )
                    
                    # Store element
                    self.staff_elements[staff_id].append(note)
                
                elif class_name == 'barline' or class_name == 'systemicBarline':
                    # Track barline positions
                    self.barlines.append((left, staff_id))
            
            return True
            
        except Exception as e:
            print(f"Error processing {xml_file}: {e}")
            return False
    
    def analyze_pitch_mapping(self):
        """
        Analyze staff elements to create a comprehensive pitch mapping
        that accounts for clefs, key signatures, and accidentals.
        """
        # Sort barlines by position
        sorted_barlines = sorted(self.barlines, key=lambda x: x[0])
        
        # Process each staff separately
        for staff_id, elements in self.staff_elements.items():
            # Ensure staff state exists
            if staff_id not in self.staff_states:
                self.staff_states[staff_id] = StaffState(staff_id=staff_id)
            
            staff_state = self.staff_states[staff_id]
            
            # Sort elements by horizontal position (left coordinate)
            sorted_elements = sorted(elements, key=lambda x: x.left)
            
            # Group elements into measures using barline positions
            measures = []
            current_measure = []
            
            current_barline_idx = 0
            for element in sorted_elements:
                # If we've passed a barline, start a new measure
                while (current_barline_idx < len(sorted_barlines) and 
                       element.left > sorted_barlines[current_barline_idx][0]):
                    if current_measure:
                        measures.append(current_measure)
                        current_measure = []
                    current_barline_idx += 1
                
                current_measure.append(element)
            
            # Add final measure
            if current_measure:
                measures.append(current_measure)
            
            # Process each measure
            for measure_idx, measure_elements in enumerate(measures):
                # Reset accidentals at each measure
                staff_state.active_accidentals = {}
                
                # Apply key signature effects
                if staff_state.key_signature:
                    # Find the KeySignature object
                    key_sig_objs = [e for e in measure_elements 
                                    if isinstance(e, KeySignature)]
                    
                    if key_sig_objs:
                        # Use the first key signature in the measure
                        key_sig = key_sig_objs[0]
                        # Apply its effects to the staff state
                        affected_pitches = key_sig.affected_pitches
                        for pitch, accidental in affected_pitches.items():
                            staff_state.active_accidentals[pitch] = accidental
                
                # Process accidentals and notes in horizontal order
                for element in measure_elements:
                    if isinstance(element, Accidental):
                        # Get the pitch this accidental affects
                        pitch_letter = self._get_pitch_at_position(
                            staff_state.clef_type, element.line_pos)
                        
                        if pitch_letter:
                            # Update active accidentals
                            if 'Sharp' in element.class_name:
                                staff_state.active_accidentals[pitch_letter] = '#'
                            elif 'Flat' in element.class_name:
                                staff_state.active_accidentals[pitch_letter] = 'b'
                            elif 'Natural' in element.class_name:
                                if pitch_letter in staff_state.active_accidentals:
                                    del staff_state.active_accidentals[pitch_letter]
                    
                    elif isinstance(element, Note):
                        # Interpret the note's pitch based on its position and active accidentals
                        interpreted_pitch = self._interpret_note_pitch(
                            element, staff_state)
                        
                        element.interpreted_pitch = interpreted_pitch
                        self.interpreted_notes.append(element)
                
                # Group notes by onset time for chord identification
                onset_groups = defaultdict(list)
                for note in [e for e in measure_elements if isinstance(e, Note)]:
                    onset_groups[note.onset_time].append(note)
                
                # For each chord (notes with same onset), sort vertically
                for onset, chord_notes in onset_groups.items():
                    if len(chord_notes) > 1:
                        # Sort notes by vertical position (top coordinate)
                        chord_notes.sort(key=lambda n: n.top)
            
        return self.interpreted_notes
    
    def _get_pitch_at_position(self, clef_type, line_pos):
        """Get the basic pitch letter for a given position and clef"""
        if not line_pos or not clef_type:
            return None
        
        # Use predefined mapping if available
        if clef_type in self.base_pitch_maps and line_pos in self.base_pitch_maps[clef_type]:
            return self.base_pitch_maps[clef_type][line_pos]
        
        # Handle positions outside the standard staff
        # Extract the position type (L or S) and number
        match = re.match(r"([LS])(-?\d+)", line_pos)
        if not match:
            return None
            
        prefix, num = match.groups()
        position_num = int(num)
        
        # Determine if we're above or below the staff
        base_map = self.base_pitch_maps.get(clef_type, {})
        
        if position_num > 5:  # Above the staff
            # Start from the pitch at L5 or S4 (whichever exists)
            if "L5" in base_map:
                base_pitch = base_map["L5"]
            elif "S4" in base_map:
                base_pitch = base_map["S4"]
            else:
                return None
                
            # Calculate distance above the staff
            if prefix == "L":
                steps_above = position_num - 5
            else:  # Space
                steps_above = position_num - 4
                
            # Cycle through pitches moving upward
            pitch_idx = "ABCDEFG".index(base_pitch)
            idx = (pitch_idx + steps_above) % 7
            return self.extensions["above"][idx]
            
        elif position_num < 1:  # Below the staff
            # Start from the pitch at L1 or S1 (whichever exists)
            if "L1" in base_map:
                base_pitch = base_map["L1"]
            elif "S1" in base_map:
                base_pitch = base_map["S1"]
            else:
                return None
                
            # Calculate distance below the staff
            if prefix == "L":
                steps_below = 1 - position_num
            else:  # Space
                steps_below = 1 - position_num
                
            # Cycle through pitches moving downward
            pitch_idx = "ABCDEFG".index(base_pitch)
            idx = (pitch_idx - steps_below) % 7
            return self.extensions["below"][idx]
        
        return None
    
    def _get_octave_at_position(self, clef_type, line_pos):
        """Get the octave for a given position and clef"""
        if not line_pos or not clef_type:
            return 4  # Default to octave 4
        
        # Use predefined mapping if available
        if clef_type in self.base_octave_maps and line_pos in self.base_octave_maps[clef_type]:
            return self.base_octave_maps[clef_type][line_pos]
        
        # Handle positions outside the standard staff
        # Extract the position type (L or S) and number
        match = re.match(r"([LS])(-?\d+)", line_pos)
        if not match:
            return 4  # Default octave
            
        prefix, num = match.groups()
        position_num = int(num)
        
        # Determine octave adjustment based on distance from staff
        base_map = self.base_octave_maps.get(clef_type, {})
        
        if position_num > 5:  # Above the staff
            # Start from the octave at L5 or S4
            if "L5" in base_map:
                base_octave = base_map["L5"]
            elif "S4" in base_map:
                base_octave = base_map["S4"]
            else:
                return 4  # Default
                
            # Calculate octave adjustment
            octave_adjustment = (position_num - 5) // 7
            if prefix == "S":
                octave_adjustment = (position_num - 4) // 7
                
            return base_octave + octave_adjustment
            
        elif position_num < 1:  # Below the staff
            # Start from the octave at L1 or S1
            if "L1" in base_map:
                base_octave = base_map["L1"]
            elif "S1" in base_map:
                base_octave = base_map["S1"]
            else:
                return 4  # Default
                
            # Calculate octave adjustment
            octave_adjustment = (1 - position_num) // 7
            if prefix == "S":
                octave_adjustment = (1 - position_num) // 7
                
            return base_octave - octave_adjustment
        
        return 4  # Default octave
    
    def _interpret_note_pitch(self, note, staff_state):
        """
        Interpret a note's pitch based on its position, clef, 
        key signature, and active accidentals
        """
        if note.midi_pitch is not None and note.pitch_step is not None and note.pitch_octave is not None:
            # If we already have complete pitch information, use it
            # But still check if there's an accidental that affects this note
            pitch_letter = note.pitch_step
            pitch_octave = note.pitch_octave
            
            # Check for active accidentals
            accidental = staff_state.active_accidentals.get(pitch_letter, "")
            
            # Form the pitch string
            pitch_str = f"{pitch_letter}{accidental}{pitch_octave}"
            return pitch_str
        
        # Otherwise, determine pitch from staff position
        if not note.line_pos or not staff_state.clef_type:
            return None
        
        # Get the basic pitch letter and octave
        pitch_letter = self._get_pitch_at_position(staff_state.clef_type, note.line_pos)
        pitch_octave = self._get_octave_at_position(staff_state.clef_type, note.line_pos)
        
        if not pitch_letter:
            return None
        
        # Check for active accidentals
        accidental = staff_state.active_accidentals.get(pitch_letter, "")
        
        # Form the pitch string
        pitch_str = f"{pitch_letter}{accidental}{pitch_octave}"
        return pitch_str
    
    def get_mapping_statistics(self):
        """
        Analyze the relationship between staff positions and interpreted pitches
        """
        # Group notes by clef, position, and interpreted pitch
        position_to_pitch = defaultdict(lambda: defaultdict(int))
        
        for note in self.interpreted_notes:
            if note.interpreted_pitch and note.line_pos:
                clef_type = self.staff_states.get(note.staff_id, StaffState(staff_id=note.staff_id)).clef_type
                key = (clef_type, note.line_pos)
                position_to_pitch[key][note.interpreted_pitch] += 1
        
        # Find most common pitch for each position
        mapping_stats = {}
        for (clef, pos), pitches in position_to_pitch.items():
            most_common_pitch = max(pitches.items(), key=lambda x: x[1])
            mapping_stats[f"{clef}_{pos}"] = {
                'clef': clef,
                'position': pos,
                'most_common_pitch': most_common_pitch[0],
                'count': most_common_pitch[1],
                'percentage': most_common_pitch[1] / sum(pitches.values()) * 100,
                'all_pitches': dict(pitches)
            }
        
        return mapping_stats

# Example usage function
def analyze_staff_position_pitch_mapping(xml_file):
    """
    Analyze the relationship between staff positions and pitches
    with consideration for clefs, key signatures, and accidentals
    """
    mapper = EnhancedPitchMapper()
    
    # Process the XML file
    success = mapper.process_xml(xml_file)
    if not success:
        print(f"Failed to process {xml_file}")
        return None
    
    # Analyze the pitch mapping
    interpreted_notes = mapper.analyze_pitch_mapping()
    
    # Get statistics on the mapping
    mapping_stats = mapper.get_mapping_statistics()
    
    print(f"\n===== ENHANCED PITCH MAPPING ANALYSIS =====")
    print(f"Total notes analyzed: {len(interpreted_notes)}")
    
    print("\nMost common pitches by position:")
    for key, stats in mapping_stats.items():
        print(f"  {stats['clef']}, {stats['position']}: "
              f"{stats['most_common_pitch']}, "
              f"{stats['count']} notes ({stats['percentage']:.1f}%)")
    
    # Return the complete mapping stats for further analysis
    return {
        'total_notes': len(interpreted_notes),
        'position_to_pitch_mapping': mapping_stats
    }
    
def main(data_directory):
    """
    Main function to run enhanced pitch mapping analysis
    
    Args:
        data_directory (str): Path to the directory containing XML files
    """
    import os
    from pathlib import Path
    import json
    from datetime import datetime
    from tqdm import tqdm
    
    print(f"Starting enhanced pitch mapping analysis on {data_directory}")
    
    # Initialize the Enhanced Pitch Mapper
    mapper = EnhancedPitchMapper()
    
    # Process all XML files in the directory
    xml_files = list(Path(data_directory).glob('**/*.xml'))
    print(f"Found {len(xml_files)} XML files")
    
    success_count = 0
    for file in tqdm(xml_files, desc="Processing XML files"):
        if mapper.process_xml(file):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(xml_files)} files")
    
    # Analyze the pitch mapping
    interpreted_notes = mapper.analyze_pitch_mapping()
    print(f"Analyzed {len(interpreted_notes)} notes")
    
    # Get statistics on the mapping
    mapping_stats = mapper.get_mapping_statistics()
    
    print(f"\n===== ENHANCED PITCH MAPPING ANALYSIS =====")
    print(f"Total notes analyzed: {len(interpreted_notes)}")
    
    print("\nMost common pitches by position:")
    for key, stats in mapping_stats.items():
        print(f"  {stats['clef']}, {stats['position']}: "
              f"{stats['most_common_pitch']}, "
              f"{stats['count']} notes ({stats['percentage']:.1f}%)")
    
    # Save the results to a JSON file
    results = {
        'timestamp': datetime.now().isoformat(),
        'data_directory': str(data_directory),
        'total_notes': len(interpreted_notes),
        'position_to_pitch_mapping': mapping_stats
    }
    
    # Custom JSON encoder for NumPy types
    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NumpyJSONEncoder, self).default(obj)
    
    # Save the results
    output_file = "enhanced_pitch_mapping_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, cls=NumpyJSONEncoder, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_file}")
    
    return results


# Call the main function
if __name__ == "__main__":
    import sys
    
    # Check if a directory is provided
    if len(sys.argv) < 2:
        print("Usage: python pitch_mapping.py <data_directory>")
        sys.exit(1)
    
    data_directory = sys.argv[1]
    main(data_directory)