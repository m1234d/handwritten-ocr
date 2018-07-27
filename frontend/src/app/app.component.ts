import { Component, OnInit, ViewChild } from '@angular/core';
import { Http, RequestOptions, Headers } from '@angular/http';
import {NgForm} from '@angular/forms';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'app';
  value = '';
  public url = 'http://localhost:5000/api/analyze';
  constructor(private http: Http) {
    this.value = '';
  }
  @ViewChild('fileInput') fileInput;

  addFile(): void {
  const fi = this.fileInput.nativeElement;
  if (fi.files && fi.files[0]) {
      const fileToUpload = fi.files[0];
      console.log('Sending file');
      this.value = 'Loading...';
      this.upload(fileToUpload)
          .subscribe(res => {
              console.log(res.json());
              this.value = 'Value: ' + res.json().value;
          });
      }
  }
  upload(fileToUpload: any) {
    const input = new FormData();
    input.append('file', fileToUpload);

    return this.http.post('http://localhost:5000/api/analyzeUpload', input);
  }

  public SendAPIRequest() {
    console.log('sending data');
    console.log(JSON.stringify(new Request()));
    const headers = new Headers({'Content-Type': 'application/json'});
    const options = new RequestOptions({ headers: headers });
    return this.http.post(this.url, JSON.stringify(new Request()), options).subscribe(result => {
      console.log(result.json());
    });
  }

}
export class Request {
  test = 'hi';
}

