import ipywidgets as widgets
import pathlib
import pandas as pd
from datetime import datetime
from IPython.display import display

class AnnotationChecker:
    """
    Check movie annotations in jupyter notebook

    Display one accordion widget per movie file containing one tab per sequence with
    fixed state (fire, confident, ...).
    In each tab, the first and last frames of the sequence are shown together with
    radio buttons containing the annotations. The user can change the annotations
    and update them by clicking on the "Save" button that creates a new csv file
    (checkedLabels.csv)

    Args:
    - inputpath: directory containing image files csv files *.labels.csv with the
      summary of movie annotations:
       * fBase: the original movie file
       * stateStart: the first frame of a sequence with fixed state
       * stateStart: the last frame of a sequence with fixed state
       * state definitions: exploitable, fire, clf_confidence, loc_confidence
       * imgFile: image file name
    - labels: csv file with labels. By default uses all files in inputdir
    """
    def __init__(self, inputpath, labels=None):
        self.inputpath = pathlib.Path(inputpath)
        assert self.inputpath.is_dir(), 'Invalid path {self.inputpath}'
        self.outputfile = self.inputpath/'checkedLabels.csv'

        if labels is None:
            csvFiles = self.inputpath.glob('*.labels.csv')
            self.labels = pd.concat(map(pd.read_csv, csvFiles))
        else:
            self.labels = pd.read_csv(labels)

        self.menus = pd.Series(index=self.labels.index)
        accordion = widgets.Accordion()
        for fBase, fgroup in self.labels.groupby('fBase'):
            tab = widgets.Tab()
            for (first, last), state in fgroup.groupby(['stateStart', 'stateEnd']):
                imgFiles = [self.inputpath/i for i in state.imgFile.iloc[[0,-1]]]
                images = [widgets.Image(value=open(imgFile, 'rb').read(), width=300)
                          for imgFile in imgFiles]
                menu = self.getMenu(state.iloc[0])
                self.menus[state.index] = menu
                hbox = widgets.HBox(images + [menu])
                tab.children += (hbox,)
                tab.set_title(len(tab.children)-1, f'Frames {int(first)}-{int(last)}')

            accordion.children += (tab,)
            accordion.set_title(len(accordion.children) - 1, fBase)
        self.accordion = accordion

        # Create 'Save' button
        self.saveButton = widgets.Button(description="Save",
                                         button_style='info',
                                         tooltip='Write results to {self.outputfile}')

        self.output = widgets.Output()
        self.saveButton.on_click(self.on_saveButton_clicked)
        self.run()


    def run(self):
        "Display widgets"
        display(self.accordion, self.saveButton, self.output)


    def getMenu(self, state):
        "Return widgets with state options"
        return widgets.VBox([
            widgets.RadioButtons(options=['Yes', 'No'],
                                 description='Usable:',
                                 value='Yes' if state['exploitable'] else 'No'),
            widgets.RadioButtons(options=['Yes', 'No'],
                                 description='Fire:',
                                 value='Yes' if state['fire'] else 'No'),
            widgets.RadioButtons(options=['Yes', 'No'],
                                 description='Confident:',
                                 value='Yes' if state['clf_confidence'] else 'No'),
            widgets.RadioButtons(options=['Precise', 'Not precise'],
                                 description='Location:',
                                 value='Precise' if state['loc_confidence'] == 2 else 'Not precise')])


    def on_saveButton_clicked(self, button):
        self.updateValues()
        self.labels.to_csv(self.outputfile, index=False)
        with self.output:
            self.output.clear_output()
            print(f"Info written to {self.outputfile} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")


    def updateValues(self):
        "Read values from menus and update self.labels"
        keys = 'exploitable', 'fire', 'clf_confidence', 'loc_confidence'
        def readValues(menu):
            values = [i.value in ('Yes', 'Precise') for i in menu.children]
            values[keys.index('loc_confidence')] += 1 # confident is 2, not-confident is 1
            return pd.Series(values, index=keys)

        self.labels.update(self.menus.apply(readValues))

